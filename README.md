# AutoAlpha v2

![AutoAlpha v2 research cockpit](autoalpha_v2/v2.png)

AutoAlpha v2 is an AI-assisted quantitative alpha research cockpit for the Scientech Labs Equity Alpha Research workflow. The project combines factor ideation, formula evaluation, knowledge-base memory, rolling model experiments, OOS combo research, and a React dashboard served by a Flask backend.

The public display build is currently served at [https://autoalpha.cn/v2](https://autoalpha.cn/v2). The display deployment uses a snapshot-only directory (`/Volumes/T7/autoalpha_v2_display`) so the website can be shown without raw parquet datasets or live mining services.

## What Is Included

- LLM-assisted factor idea generation and prompt memory.
- Formula parsing, leakage checks, and submission-like evaluation.
- A local knowledge base of tested factors, gate status, scores, IC/IR/TVR, and research notes.
- Rolling model lab and exploratory OOS combo lab.
- Full-factor and low-correlation 8-factor combo comparisons on 2024 OOS data.
- ML benchmarks over raw/rank/z-score factor features: Ridge, RandomForest, ExtraTrees, HistGradientBoosting, LightGBM, and MLP.
- React + Recharts frontend for research monitoring, factor records, inspirations, and combo cards.
- Display-only Flask server that serves precomputed JSON snapshots and static assets.

## Latest OOS Combo Snapshot

The latest computed exploratory combo lab uses a chronological split:

- Train: `2022-01-04` to `2023-12-29`
- Validation: last two visible train months inside 2022-2023
- Test: `2024-01-02` to `2024-12-31`
- No 2024 labels are used for fitting weights, model parameters, method selection, or validation.

| Lab | Best model | Factor set | 2024 OOS Score | IC | IR | TVR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full factor combo | `RidgeZScoreMetaModel` | 96 | 8659.93 | 12.12 | 51.61 | 136.08 |
| Low-correlation combo | `LightGBMMetaModel` | 8 | 4399.31 | 7.91 | 31.48 | 136.28 |

Each combo card in the UI includes the method description, leakage guard, Train/Val/OOS metric comparison, TVR, and OOS time-series comparison.

## Repository Layout

```text
.
├── autoalpha_v2/              # AutoAlpha v2 research package and runtime state
│   ├── rolling_model_lab.py   # OOS combo lab and ML/meta-model experiments
│   ├── pipeline.py            # idea -> formula -> evaluate workflow
│   ├── knowledge_base.py      # factor knowledge-base persistence
│   ├── llm_client.py          # LLM routing and prompt calls
│   ├── inspiration_db.py      # inspiration/prompt database utilities
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

## Runtime Directories

The project has two local runtime directories:

| Directory | Purpose |
| --- | --- |
| `/Volumes/T7/autoalpha_v2` | Full research workspace. Can run mining, evaluation, model lab, and frontend/backend. |
| `/Volumes/T7/autoalpha_v2_display` | Display-only deployment. Contains built frontend, compact JSON snapshots, and selected display parquet outputs only. |

The display directory does not import the research package and does not need raw parquet files. It serves precomputed snapshots for faster public access.

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
- Public tunnel target: `https://autoalpha.cn/v2`

## Running The Combo Lab

Full-factor lab:

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
- Factor records
- Inspiration browser
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
4. Copy selected display outputs to `/Volumes/T7/autoalpha_v2_display/data/submit`.
5. Restart `/Volumes/T7/autoalpha_v2_display/start_display.sh`.

The snapshot API is intentionally read-only. Mutating endpoints return `403` in display mode.

## License And Competition Data

The code can be shared in the project repository. Competition data, generated parquet outputs, credentials, and local databases remain outside Git because they may be large, private, or environment-specific.
