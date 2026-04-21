# AutoAlpha v1

AutoAlpha is the factor research pipeline used in this workspace. It generates
intraday alpha ideas, validates formulas, computes 15-minute alpha files,
evaluates them with an official-like metric implementation, and keeps a
knowledge base plus submit-ready artifacts for factors that pass all gates.

## What v1 Fixes

The v1 branch aligns factor export and metric calculation with the platform
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
`autoalpha/output/debug` and the manual submit result JSON files in
`manual/submit` and `submit`.

## Directory Layout

```text
autoalpha/
├── llm_client.py              # LLM idea generation
├── pipeline.py                # generate -> validate -> compute -> evaluate -> export
├── run.py                     # CLI entry point for new factor generation
├── recompute_gate_factors.py  # rebuild prior gate-passing factors with v1 metrics
├── knowledge.json             # factor registry used by the frontend
├── output/                    # regenerated parquet files and run manifests
├── submit/                    # submit-ready passing factors and metadata
├── submit_repaired/           # repaired reference submit candidates
├── research/                  # per-factor LOG reports
└── recompute_reports/         # batch recomputation summaries
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
python autoalpha/run.py --n 3

# Generate with a shorter evaluation window for faster iteration.
python autoalpha/run.py --n 3 --days 120
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
7. Passing factors are copied to `autoalpha/submit` with metadata and an
   official-like result JSON.

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
python autoalpha/recompute_gate_factors.py

# Faster: keep existing LOG reports and only refresh pq + metrics.
python autoalpha/recompute_gate_factors.py --skip-research

# Recompute every factor in knowledge.json.
python autoalpha/recompute_gate_factors.py --all --skip-research
```

The script:

- backs up `autoalpha/knowledge.json`;
- archives the previous `autoalpha/submit` contents;
- recomputes each formula from source data;
- regenerates `autoalpha/output/<run_id>.pq`;
- refreshes `knowledge.json` metrics and gate fields;
- copies only still-passing factors into `autoalpha/submit`;
- writes a batch summary under `autoalpha/recompute_reports`.

Submit-ready factors are the `.pq` files directly under `autoalpha/submit`.
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

## Useful Commands

```bash
# Check Python syntax for the recompute script.
python -m py_compile autoalpha/recompute_gate_factors.py

# Build the frontend.
npm --prefix frontend run build

# Inspect current submit candidates.
find autoalpha/submit -maxdepth 1 -name '*.pq' -print

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
