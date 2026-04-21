# AutoAlpha v2

This is a clean code-only copy for restarting AutoAlpha from scratch. Runtime
state and generated artifacts are intentionally excluded: no `knowledge.json`,
no parquet outputs, no research reports, no submit files, no model cache, and no
local database.

AutoAlpha is the factor research pipeline used in this workspace. It generates
intraday alpha ideas, validates formulas, computes 15-minute alpha files,
evaluates them with an official-like metric implementation, and keeps a
knowledge base plus submit-ready artifacts for factors that pass all gates.

## v2 Technical Idea

AutoAlpha v2 is a closed-loop intraday factor factory. The core idea is to let
agents continuously propose compact DSL formulas, then force every formula
through the same deterministic chain: syntax/compliance validation, warmup-aware
factor computation, official-like metric evaluation, submission-grid export,
research diagnostics, knowledge-base persistence, and parent selection for the
next generation.

The pipeline deliberately separates **idea quality** from **submission
readiness**:

- Idea agents only output `formula`, `thought_process`, `postprocess`, and
  `lookback_days`. They do not directly write files or decide whether a factor
  is good.
- `formula_validator.py` and the operator registry define the legal search
  space, blocking unsupported operators and future-information leaks.
- `pipeline.compute_alpha` loads warmup history before the evaluation window so
  rolling operators are initialized correctly, then trims back to the true
  evaluation days.
- Recent-window screening avoids spending full-history compute on obviously weak
  ideas, while promising ideas are recomputed on the full available period.
- `core.evaluator.evaluate_submission_like_wide` is the single source of truth
  for IC, IR, turnover, concentration, coverage, gate flags, and score.
- Only factors that pass all gates are copied into `autoalpha_v2/submit` and shown
  as submit-ready card links in the research UI.
- Passing factors become parents and examples for later agents through
  `knowledge.json`, structural fingerprints, operator-pair memory, and the
  frontend records page.

In practice v2 behaves like an autonomous research desk: agents create
hypotheses, the platform-grade evaluator acts as a hard reviewer, and the
knowledge layer remembers both productive motifs and exhausted formula families.

## Difference From v0

The repository root still contains the earlier v0-style research stack described
in `/Volumes/T7/Scientech/README.md`: manual/EA/LLM scripts, general DSL
parsing, local leaderboard tracking, backtest helpers, and submission utilities.
AutoAlpha v2 is narrower and stricter. It keeps the useful v0 infrastructure but
wraps it in an operational loop whose outputs are immediately auditable and
submit-ready.

| Area | v0 root workflow | AutoAlpha v2 |
|------|------------------|--------------|
| Research mode | Script-oriented experiments: `research_loop.py`, `evaluate_alpha.py`, manual configs, leaderboard updates. | Productized loop: `autoalpha_v2/run.py` / `loop.py` generate, screen, export, report, notify, and persist every result. |
| Metric alignment | Useful local metrics existed, but older exports could diverge from platform assumptions. | Official-like 15-minute evaluator, post-restriction metrics, corrected TVR, concentration gates, and full-grid parquet checks are mandatory. |
| Artifact policy | Many exploratory outputs live under `outputs/`, `research/`, `submit/`, and manual reports. | Passing factors get canonical `.pq`, metadata, official-like result JSON, report, and factor card under `autoalpha_v2/submit` and `autoalpha_v2/research`. |
| Knowledge memory | Leaderboard and logs guide later iterations informally. | `knowledge.json` stores every tested factor, failure reason, parent lineage, fingerprints, card paths, lab-test results, and generation summaries. |
| Agent feedback | Top formulas can be reused, but failure families are less explicit. | LLM prompts receive strong examples, recent weak examples, productive operator pairs, and saturated structural families. |
| Frontend | General dashboard/backend integration. | Dedicated AutoAlpha cockpit: quota/status, prompt lab, rolling model lab, generation records, submit-card links, and inspiration database. |
| Submission safety | Submission helpers exist but can be called independently. | Submit readiness is a gate-controlled state; only `PassGates=true` factors are copied to `autoalpha_v2/submit` and rendered with factor cards. |

## What v2 Keeps From v1

The v2 branch keeps the v1 factor export and metric calculation alignment with the platform
rules:

- Alpha parquet export now normalizes `date`, `datetime`, and `security_id`
  before building the full submission grid. This prevents the all-null / tiny
  file failure mode where generated factors looked valid locally but exported
  only about 80 MB of useful data.
- Metrics are computed after applying `trading_restriction`, matching the
  platform description: restricted securities are removed before IC, turnover,
  book weights, and concentration are calculated.
- Turnover (`tvr`) is the mean daily sum of per-bar raw alpha changes,
  normalized by current absolute alpha magnitude and reported as percentage
  points (`x100`).
- Position metrics use the same weight idea as the reference text:
  `10000 * alpha_i / sum(abs(alpha))`. Negative values are kept negative for
  `bs`, `minn`, and `min`.
- Gate and score logic follows the reference:
  `cover_all = 1`, `IC > 0.6`, `IR > 2.5`, `tvr < 400`,
  `maxx < 50`, `abs(minn) < 50`, `max < 20`, `abs(min) < 20`, then
  `score = (IC - 0.0005 * tvr) * sqrt(IR) * 100`.

The known alignment references are the repaired factors under
`autoalpha_v2/output/debug` and the manual submit result JSON files in
`manual/submit` and `submit`.

## Directory Layout

```text
autoalpha_v2/
├── llm_client.py              # LLM idea generation
├── pipeline.py                # generate -> validate -> compute -> evaluate -> export
├── run.py                     # CLI entry point for new factor generation
├── recompute_gate_factors.py  # rebuild prior gate-passing factors with current metrics
└── .gitignore                 # keeps regenerated runtime artifacts out of git
```

Project-level files used by AutoAlpha:

```text
core/evaluator.py      # official-like metric implementation
core/submission.py     # full-grid parquet export helpers
prepare_data.py        # DataHub loader for pv, response, and restrictions
start_all.sh           # backend/frontend launcher
frontend/              # React UI, including AutoAlpha records
```

## Quick Start

From the repository root:

```bash
cd /Volumes/T7/Scientech

# Start backend and frontend.
./start_all.sh

# Reuse already healthy services.
./start_all.sh --reuse

# Generate new factors on the full available evaluation period.
python autoalpha_v2/run.py --n 3

# Generate with a shorter evaluation window for faster iteration.
python autoalpha_v2/run.py --n 3 --days 120
```

The launcher starts:

- backend: `http://127.0.0.1:8080`
- frontend: `http://127.0.0.1:3000`

On macOS it uses `launchctl` LaunchAgents so the services survive the terminal
command that started them. Logs are written to `~/Library/Logs/Scientech`.

## Factor Generation Flow

1. `llm_client.py` asks the LLM for an idea with `formula`,
   `thought_process`, `postprocess`, and `lookback_days`.
2. `formula_validator.py` rejects unsupported fields/operators and obvious
   future-information leaks.
3. `pipeline.compute_alpha` evaluates the formula on `DataHub.pv_15m`, loading
   warmup history before the evaluation window so time-series operators have
   enough past data.
4. Post-processing converts raw signal values into a submit-friendly cross
   section, usually `rank_clip` in `[-0.5, 0.5]` or clipped z-score variants.
5. `pipeline.evaluate_alpha` calls `core.evaluator.evaluate_submission_like_wide`
   using response data and trading restrictions.
6. `pipeline.export_parquet` writes the normalized full-grid `.pq` file with
   columns `date`, `datetime`, `security_id`, and `alpha`.
7. `factor_research.analyze_factor` builds a research report. If and only if the
   factor passes all submit gates, it also writes `factor_card.json` and
   `factor_card.md`.
8. Passing factors are copied to `autoalpha_v2/submit` with metadata and an
   official-like result JSON; their `run_id` becomes a clickable card link in
   the research records table.

## Factor Cards

Factor cards are only generated for submit-ready factors (`PassGates=true`).
Rejected, duplicate, invalid, compute-error, and screened-out factors remain in
the research log and table, but they do not get a factor card. This keeps the
card library focused on candidates that can actually be submitted.

Each card is stored beside the factor report:

```text
autoalpha_v2/research/<run_id>/
├── report.json
├── report.md
├── analysis.png
├── factor_card.json
└── factor_card.md
```

The frontend reads `factor_card_path` from `knowledge.json`. In the knowledge
table, only submit-ready factors display the `run_id` as a link; clicking it
opens the card/report modal. Numeric card metrics are displayed with four
decimal places so the UI stays readable.

The card covers eight compact but useful views:

| Section | What it shows | Why it matters |
|---------|---------------|----------------|
| Factor definition | Formula, input fields, update frequency, prediction horizon, universe, postprocess. | Explains what the factor captures and helps deduplicate ideas. |
| Historical distribution | Histogram, P1/P5/P50/P95/P99, mean, std, skew, kurtosis, missing rate, extreme share. | Reveals skew, outlier dependence, and whether clip/rank/zscore is needed. |
| Temporal evolution | Daily mean, daily std, coverage, rolling drift. | Shows drift, unstable regimes, and coverage breaks. |
| Predictive power | IC mean, ICIR, Rank IC, rolling IC, horizon/lag IC. | Answers whether the factor works and at which horizon. |
| Layered performance | Decile return bars, top-minus-bottom spread, cumulative spread curve. | Checks monotonicity and whether only tails work. |
| Good regimes | IC by high/low volatility, trend/chop, or available response regimes. | Identifies when the factor should be enabled or gated. |
| Stability | Monthly/yearly IC, train/val/test split, clipped-tail IC. | Tests whether performance is period- or outlier-dependent. |
| Correlation and redundancy | Formula-family label, nearest known factor proxy, alpha-pool overlap proxy, target-correlation proxy. | Keeps the alpha pool useful rather than repetitive. |

For future research runs the card flow is automatic: once full-history metrics
return `PassGates=true`, `analyze_factor` writes the card, `knowledge_base`
records the path, and the frontend makes the factor name clickable.

## DSL Formula Language

The v1 DSL is intentionally restricted to current/past 15-minute bar data, but
the operator set is now broad enough for richer factor search.

Fields:

```text
open_trade_px, high_trade_px, low_trade_px, close_trade_px,
trade_count, volume, dvolume, vwap
```

Time-series operators:

```text
lag(x,d), delay(x,d), delta(x,d), ts_pct_change(x,d),
ts_mean(x,d), ts_ema(x,d), ts_std(x,d), ts_sum(x,d),
ts_max(x,d), ts_min(x,d), ts_median(x,d), ts_quantile(x,d,q),
ts_zscore(x,d), ts_rank(x,d), ts_minmax_norm(x,d),
ts_decay_linear(x,d), decay_linear(x,d),
ts_corr(x,y,d), ts_cov(x,y,d),
ts_skew(x,d), ts_kurt(x,d), ts_argmax(x,d), ts_argmin(x,d)
```

Cross-sectional operators:

```text
cs_rank(x), rank(x), cs_zscore(x), zscore(x),
cs_demean(x), demean(x), cs_scale(x), scale(x),
cs_winsorize(x,p), winsorize(x,p), cs_quantile(x,q),
cs_neutralize(x,y)
```

Math, condition, and blend operators:

```text
safe_div(a,b), div(a,b), signed_power(x,p), pow(x,p),
abs(x), sign(x), neg(x), log(x), signed_log(x), sqrt(x),
clip(x,a,b), clamp(x,a,b), min_of(x,y), max_of(x,y),
sigmoid(x), tanh(x),
ifelse(cond,a,b), gt(x,y), ge(x,y), lt(x,y), le(x,y), eq(x,y),
and_op(a,b), or_op(a,b), not_op(a),
mean_of(x1,x2,...), weighted_sum(w1,x1,w2,x2,...),
combine_rank(x1,x2,...)
```

Useful new formula motifs:

```text
# Robust VWAP dislocation with median baseline
cs_zscore(ts_decay_linear(safe_div(close_trade_px - vwap, ts_median(high_trade_px - low_trade_px, 16)), 6))

# Liquidity-neutral short-term momentum
cs_neutralize(ts_zscore(ts_pct_change(close_trade_px, 2), 20), ts_zscore(volume, 20))

# Soft regime gate instead of a hard if/else
tanh(ts_zscore(delta(close_trade_px, 1), 20)) * sigmoid(ts_zscore(delta(trade_count, 1), 12))

# Multi-leg blend
combine_rank(neg(ts_zscore(close_trade_px - vwap, 16)), ts_minmax_norm(trade_count, 20))
```

Safety rules:

- Lookback arguments such as `d` must be positive integer literals.
- `lead`, `future_*`, `resp`, and `trading_restriction` are forbidden.
- Conditional operators are allowed, but they can raise turnover; smooth the
  result with `ts_mean`, `ts_ema`, or `ts_decay_linear` when possible.
- Prefer `safe_div` for series denominators and infix `/` for simple scalar
  constants.

## Official-Like Metrics

The evaluator runs on the platform cadence of 15-minute bars. If an alpha was
computed at a finer frequency, only the final value in each platform bar should
be used before evaluation.

The main reported metrics are:

- `IC`: mean daily cross-sectional Pearson correlation with response, multiplied
  by 100.
- `IR`: annualized consistency of daily IC,
  `mean(daily_ic) / std(daily_ic) * sqrt(252)`.
- `tvr`: mean daily turnover. Per bar:
  `sum(abs(alpha_t - alpha_t-1)) / sum(abs(alpha_t))`; daily turnover is the
  sum over bars, and the displayed value is `x100`.
- `bl` / `bs`: mean total long and short book weight per bar.
- `nl` / `ns` / `nt`: mean counts of long, short, and non-zero alpha names per
  bar.
- `maxx` / `minn`: largest positive and most negative single-security position
  weight in bps across all bars.
- `max` / `min`: daily mean of maximum and minimum single-security position
  weight in bps.
- `nd`: number of trading days evaluated.
- `cover_all`: 1 when every evaluation day is covered.

Quality gates and score are stored in both `knowledge.json` and submit metadata.
The frontend reads these values directly, so any metric change must go through
the recomputation path below.

## Recomputing Submit Candidates

When metric/export logic changes, rebuild prior candidates instead of trusting
old `knowledge.json` values:

```bash
# Recompute only factors that previously had PassGates=true.
python autoalpha_v2/recompute_gate_factors.py

# Faster: keep existing LOG reports and only refresh pq + metrics.
python autoalpha_v2/recompute_gate_factors.py --skip-research

# Recompute every factor in knowledge.json.
python autoalpha_v2/recompute_gate_factors.py --all --skip-research
```

The script:

- backs up `autoalpha_v2/knowledge.json`;
- archives the previous `autoalpha_v2/submit` contents;
- recomputes each formula from source data;
- regenerates `autoalpha_v2/output/<run_id>.pq`;
- refreshes `knowledge.json` metrics and gate fields;
- copies only still-passing factors into `autoalpha_v2/submit`;
- writes a batch summary under `autoalpha_v2/recompute_reports`.

Submit-ready factors are the `.pq` files directly under `autoalpha_v2/submit`.
Each has a sibling metadata JSON and official-like result JSON that records the
exact metrics used by the UI.

## Frontend Records

The AutoAlpha records page shows:

- generation lineage;
- output and LOG artifacts;
- the factor table with `Status/Gate`, `Lab Test`, and `LOG` columns;
- distinct row colors for failure reasons, unsubmitted passing factors, and
  factors with filled Lab Test results.

Lab Test results can be pasted into the row modal. They are stored back into the
knowledge base and displayed separately from local official-like metrics.

## Frontend Examples

AutoAlpha v1 includes three main frontend surfaces.

![AutoAlpha cockpit](docs/images/frontend-autoalpha.png)

![AutoAlpha research records](docs/images/frontend-records.png)

![AutoAlpha inspiration library](docs/images/frontend-inspirations.png)

## Useful Commands

```bash
# Check Python syntax for the recompute script.
python -m py_compile autoalpha_v2/recompute_gate_factors.py

# Build the frontend.
npm --prefix frontend run build

# Inspect current submit candidates.
find autoalpha_v2/submit -maxdepth 1 -name '*.pq' -print

# Start services after a recompute.
./start_all.sh
```

## Operational Notes

- Generated parquet files can be large and are treated as runtime artifacts.
  Keep code, metadata, README, and recompute summaries in Git; avoid committing
  bulk `.pq` outputs unless a release explicitly needs binary artifacts.
- `AUTOALPHA_CLOUD_TVR_MULTIPLIER` can be used to apply a conservative turnover
  multiplier in local scoring, but v1 defaults to `1.0` after aligning turnover
  to the official-like calculation.
- If a factor scores well locally but has `score = 0` in Lab Test, compare its
  `*_official_like_result.json`, parquet file size, and `cover_all` first. The
  most common failure modes are incomplete export grids and stale metric data.
