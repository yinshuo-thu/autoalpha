# AutoAlpha v2

![AutoAlpha v2 research cockpit](autoalpha_v2/v2.png)

AutoAlpha v2 is an AI-assisted intraday alpha research factory for the Scientech Labs Equity Alpha Research workflow. It is not only a combo-result dashboard: the core of the project is a closed-loop factor mining system that turns research hypotheses into DSL formulas, validates them, evaluates them on 15-minute data, stores the research memory, and then studies single-factor and multi-factor OOS behavior.

The server-hosted display frontend is available at [https://autoalpha.cn/v2](https://autoalpha.cn/v2). This website is a public showcase frontend deployed on a server. It does not provide raw parquet downloads, but it does expose the full mining process, factor ideas, factor records, generation history, RAG context, and combo-lab summaries through compact snapshots.

## Research Goal

AutoAlpha v2 was built around one practical question:

> Can an agentic research loop discover, test, remember, and combine intraday alpha factors with enough structure and auditability to be useful for real quant research?

The current implementation focuses on:

- generating interpretable factor hypotheses from manual prompts, imported research notes, and LLM reasoning;
- translating hypotheses into a constrained DSL formula language;
- parsing formulas into ASTs so the system can inspect fields, operators, windows, and structure;
- rejecting leakage-prone or invalid formulas before expensive evaluation;
- computing factor files and official-like metrics including IC, IR, turnover/TVR, concentration, coverage, and score;
- saving passing and failing attempts into a knowledge base for later retrieval;
- using RAG and generation experience to steer the next mining rounds;
- combining mined factors with rank-based ensembles and ML meta-models under chronological train/validation/OOS splits;
- serving a compact, read-only public frontend without exposing private raw data.

## End-to-End Mining Loop

The v2 loop is designed as a research factory rather than a single model call.

1. **Inspiration intake**
   - Manual factor prompts, notes under `manual/`, futures/market microstructure notes under `fut_feat/`, and prior successful factors are converted into structured inspiration records.
   - Each inspiration can be summarized, tagged, sampled, and later tied back to generated factors.

2. **Hypothesis and formula generation**
   - `autoalpha_v2/llm_client.py`, `autoalpha_v2/pipeline.py`, and `autoalpha_v2/loop.py` generate research hypotheses and DSL formulas.
   - Prompts include the target metric, allowed fields/operators, known good structures, failure feedback, and selected RAG context.

3. **DSL parsing and structural control**
   - Formulas are parsed into ASTs by `formula_parser.py`.
   - The system can collect fields, operators, numeric windows, structural fingerprints, and parent-child formula relationships.
   - This makes factors auditable as formula trees instead of opaque strings.

4. **Validation and leakage guard**
   - `formula_validator.py` and `compliance_guard.py` enforce syntax validity, operator whitelists, field whitelists, no `resp` usage, no `trading_restriction` usage in factor construction, and bounded output behavior.

5. **Evaluation and gate checks**
   - `quick_test.py`, `core/evaluator.py`, and `core/submission.py` compute factor values, apply post-processing, evaluate IC/IR/TVR/concentration, and package passing factors.
   - Factors that pass gates are copied to submit-ready outputs; runtime parquet files remain outside Git.

6. **Memory update**
   - `autoalpha_v2/knowledge_base.py` records formula, metrics, generation, parents, inspiration IDs, structural fingerprint, status, and research paths.
   - Passing factors become future RAG anchors; failed families can be down-weighted or treated as exhausted.

7. **Model and combo lab**
   - `autoalpha_v2/rolling_model_lab.py` studies chronological OOS behavior, low-correlation factor subsets, rank ensembles, and ML meta-models.
   - Results are exported as compact JSON summaries for the frontend and full runtime artifacts for local research.

## RAG And Research Memory

The RAG layer is intentionally research-oriented. It is not a generic document chat system; it retrieves compact evidence that can change the next factor generation step.

Current memory sources include:

- passing factor records with formula, score, IC, IR, TVR, generation, parent IDs, and thought process;
- failed or exhausted structural families through formula fingerprints;
- recent generation summaries under `autoalpha_v2/generation_notes/`;
- inspiration records from manual prompts and imported notes;
- leaderboard-style top factors used as strong anchors;
- combo-lab summaries used to understand factor complementarity and redundancy.

The project also documents planned RAG upgrades in `autoalpha_v2/RAG_TODO.md`, including semantic retrieval for passing factors, dynamic inspiration quality feedback, finer-grained structural fingerprints, historical experience retrieval, and Stage-1 hypothesis outcome feedback.

## Compact Display Architecture

Raw parquet files and local databases are large and private, so the public site uses a compact deployment design.

- The full research workspace lives at `/Volumes/T7/autoalpha_v2`.
- The display deployment lives at `/Volumes/T7/autoalpha_v2_display`.
- `server.py` compacts heavy model-lab summaries by keeping public-safe metadata, selected metrics, method cards, small curves, correlations, and display records while stripping raw formulas or bulky arrays where needed.
- The display server serves built frontend assets and read-only JSON snapshots.
- Mutating endpoints are disabled in display mode.

This is why [https://autoalpha.cn/v2](https://autoalpha.cn/v2) should be described as a deployed showcase frontend, not as a raw-data download portal. The website is meant to let visitors inspect the mining process and factor research logic without exposing original parquet data.

## Current Mining Results

As of the latest local snapshot on 2026-04-26:

| Area | Result |
| --- | ---: |
| Total tested factor records | 1,998 |
| Passing factors in AutoAlpha KB | 96 |
| Main generations with passing factors | generation 7 to 17 |
| Best single-factor Score | 209.28 |
| Best single-factor IC | 1.1478 |
| Best single-factor IR | 4.2562 |
| Best single-factor TVR | 266.75 |

Representative top single factor:

```text
autoalpha_20260422_223541_02
Score 209.28 | IC 1.1478 | IR 4.2562 | TVR 266.75
```

Research interpretation: a short-horizon continuation signal that requires positive 4-bar price movement, recent range location near the high, and participation expansion confirmed by both volume and trade count. The outer decay is used to reduce turnover while preserving intraday persistence.

## Latest OOS Combo Snapshot

The latest computed combo labs use chronological splits:

- Train: `2022-01-04` to `2023-12-29`
- Validation: only inside the 2022-2023 training period
- Mock OOS test: `2024-01-02` to `2024-12-31`
- No 2024 labels are used for fitting weights, model parameters, method selection, or validation.

| Lab | Best model | Factor set | 2024 OOS Score | IC | IR | TVR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full factor combo | `RidgeZScoreMetaModel` | 96 | 8659.93 | 12.12 | 51.61 | 136.08 |
| Low-correlation combo | `LightGBMMetaModel` | 8 | 4399.31 | 7.91 | 31.48 | 136.28 |

The combo dashboard includes method cards, leakage guards, Train/Val/OOS metric comparison, TVR curves, feature contribution, factor correlation views, and OOS time-series comparison.

## What Is Included

- LLM-assisted factor idea generation and prompt memory.
- DSL formula parsing, AST inspection, formula fingerprints, and structure-aware mutation.
- Leakage and compliance checks before evaluation.
- Local knowledge base for tested factors, gate status, scores, IC/IR/TVR, parentage, inspiration source, and research notes.
- RAG context for passing factors, recent generation experience, and inspiration records.
- Rolling model lab and exploratory OOS combo lab.
- Full-factor and low-correlation combo comparisons on 2024 mock OOS data.
- ML benchmarks over raw/rank/z-score factor features: Ridge, RandomForest, ExtraTrees, HistGradientBoosting, LightGBM, and MLP.
- React + Recharts frontend for mining progress, factor records, inspirations, RAG roadmap, and combo cards.
- Display-only Flask server for compact, public-safe JSON snapshots and static assets.

## Repository Layout

```text
.
├── autoalpha_v2/              # AutoAlpha v2 research package and runtime state
│   ├── loop.py                # closed-loop factor mining orchestration
│   ├── pipeline.py            # idea -> formula -> evaluate workflow
│   ├── llm_client.py          # LLM routing and prompt construction
│   ├── knowledge_base.py      # factor KB, RAG context, fingerprints
│   ├── inspiration_db.py      # inspiration/prompt database utilities
│   ├── rolling_model_lab.py   # OOS combo lab and ML/meta-model experiments
│   ├── RAG_TODO.md            # RAG improvement roadmap
│   └── v2.png                 # README hero image
├── core/                      # data loading, evaluator, submission utilities
├── factors/                   # factor formula library and prompts
├── frontend/                  # React/Vite dashboard
├── fut_feat/                  # imported factor inspiration notes
├── manual/                    # manual prompt/research artifacts
├── research/                  # notebooks/configs/research notes
├── scripts/                   # maintenance and snapshot helper scripts
├── server.py                  # live Flask API + frontend server
├── runtime_config.py          # runtime config loader/saver
├── prepare_data.py            # data hub and raw-data alignment entrypoint
└── requirements.txt           # Python dependency baseline
```

Large runtime outputs are intentionally excluded from Git:

- raw and derived parquet files (`*.pq`, `*.parquet`)
- local SQLite databases (`*.db`, `*.sqlite*`)
- generated submit/output/model-lab artifacts
- frontend `node_modules` and `dist`
- logs, pid files, Python caches, macOS AppleDouble files

## Quick Start

### 1. Python environment

```bash
cd /Volumes/T7/autoalpha_v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The local machine currently uses Conda Python for large experiments. If using Conda:

```bash
conda activate <your-env>
pip install -r requirements.txt
```

### 2. Frontend dependencies

```bash
cd /Volumes/T7/autoalpha_v2/frontend
npm install
npm run build
```

### 3. Live research server

```bash
cd /Volumes/T7/autoalpha_v2
python server.py
```

The live server defaults to:

- Backend/API: `http://127.0.0.1:8080`
- Frontend path: `http://127.0.0.1:8080/v2/`

### 4. Display-only server

```bash
cd /Volumes/T7/autoalpha_v2_display
./start_display.sh
```

The display server is read-only and serves:

- Local display: `http://127.0.0.1:8080/v2/`
- Public display frontend: `https://autoalpha.cn/v2`

## Running Research Jobs

Closed-loop mining:

```bash
PYTHONPATH=/Volumes/T7/autoalpha_v2 python -m autoalpha_v2.run
```

Full-factor combo lab:

```bash
PYTHONPATH=/Volumes/T7/autoalpha_v2 python -m autoalpha_v2.rolling_model_lab \
  --target-valid 96 \
  --ideas-per-round 0 \
  --max-rounds 0 \
  --allow-partial
```

Low-correlation 8-factor lab:

```bash
PYTHONPATH=/Volumes/T7/autoalpha_v2 python -m autoalpha_v2.rolling_model_lab \
  --run-low-corr-experiment
```

The lab exports compact summaries under `autoalpha_v2/model_lab/` and submit-ready outputs under `autoalpha_v2/submit/`. These runtime artifacts are excluded from source control.

## Frontend Notes

The dashboard is built with:

- React 18
- TypeScript
- Vite
- Tailwind CSS
- Recharts
- lucide-react

The UI is organized around:

- AutoAlpha Research Cockpit
- Prompt Lab
- Loop control and live logs
- Factor records and factor cards
- Inspiration browser
- RAG roadmap and development timeline
- Exploratory OOS Combo Lab
- Combo Card drilldowns

For production builds under `/v2`, `frontend/vite.config.ts` reads `AUTOALPHA_APP_BASE` and defaults to `/v2/`.

## Data And Secret Policy

This repository is meant to publish project code, not private runtime data. Do not commit:

- raw market parquet files
- generated submission parquet files
- local SQLite databases
- API keys or `.env` files
- process logs with private endpoints
- `node_modules`

Before publishing, run:

```bash
git status --short
git ls-files | rg '(\.pq$|\.parquet$|\.db$|\.sqlite|\.env|node_modules|__pycache__)'
```

The expected result for the second command is empty.

## Deployment Snapshot Workflow

1. Build frontend in `/Volumes/T7/autoalpha_v2/frontend`.
2. Sync `frontend/dist` to `/Volumes/T7/autoalpha_v2_display/frontend/dist`.
3. Write compact JSON snapshots to `/Volumes/T7/autoalpha_v2_display/data/snapshots`.
4. Copy selected display-safe outputs to `/Volumes/T7/autoalpha_v2_display/data/submit`.
5. Restart `/Volumes/T7/autoalpha_v2_display/start_display.sh`.

The snapshot API is intentionally read-only. Mutating endpoints return `403` in display mode.

## Contact

I welcome conversations about this project, recent progress, collaboration, and internship opportunities.

- Email: [yinelon@gmail.com](mailto:yinelon@gmail.com)
- LinkedIn: [Shuo Yin](https://www.linkedin.com/in/shuoyin/)

## License And Competition Data

The code can be shared in the project repository. Competition data, generated parquet outputs, credentials, and local databases remain outside Git because they may be large, private, or environment-specific.
