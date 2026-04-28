# AutoAlpha v3 Directory Guide

This branch is organized as a code-first archive. Runtime outputs and private
data are intentionally excluded; the tracked files should be enough to migrate,
review, rebuild the frontend, and resume development.

## Runtime Core

- `loop.py` - closed-loop factor mining orchestration.
- `pipeline.py` - idea generation, validation, factor computation, evaluation,
  export, and research-card generation.
- `run.py` - CLI entrypoint for factor generation.
- `server.py` - Flask API and frontend static server.
- `rolling_model_lab.py` - OOS combo lab, low-correlation basket tests, ML/DL
  model comparisons, and fusion diagnostics.
- `runtime_config.py`, `paths.py` - runtime configuration and shared paths.

## Libraries

- `core/` - evaluator, submission writer, data helpers, formula engine, and
  shared mining utilities.
- `factors/` - factor operators and prompt materials.
- `research/` - research runner, configs, score-formula analysis, and archived
  lightweight run snapshots under `research/runs/`.
- `manual/` - curated manual factor prompts and manual-factor tooling.

## Frontend And Deployment

- `frontend/` - React/Vite dashboard and lightweight backend helpers.
- `deploy/` - display-only server and deployment scripts.
- `scripts/` - operational scripts for rolling labs, systematic mining, v2
  imports, and LLM factor runs.
- `outputs/export_submission.py` - retained source helper; generated output
  files under `outputs/` remain ignored.

## Documentation And Archive Material

- `docs/requirements/` - project requirements and task framing.
- `docs/rag/` - RAG roadmap and implementation notes.
- `docs/notes/generation/` - archived generation summaries.
- `docs/summaries/` - lightweight research summaries.
- `assets/images/` - documentation images.
- `examples/legacy_factor_research/数据及因子/` - legacy/reference factor
  implementations kept for migration and comparison, not part of the main
  runtime import path.

## Ignored Runtime State

The following are intentionally not tracked: raw parquet data, local databases,
`output/`, `submit/`, `model_lab/`, process logs, frontend builds,
`node_modules`, Python caches, and generated research directories such as
`research/autoalpha_*`.
