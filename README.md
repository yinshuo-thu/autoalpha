# Scientech Alpha Research Factory

An autonomous quantitative alpha research system built for the **Scientech Labs Equity Alpha Research 2026** competition. It covers the full pipeline: data ingestion, factor generation, compliance checking, backtest evaluation, and submission packaging.

## Competition Overview

| Item | Detail |
|------|--------|
| Data | 1-min OHLCV bars (2022–2024), resampled to 15-min |
| Submission format | 15-min frequency alpha grid (Parquet) |
| Scoring | `score = (IC - 0.0005 * tvr) * sqrt(IR) * 100` |
| IC target | > 0.6 (scaled ×100) |
| IR target | > 2.5 |
| Turnover target | tvr < 400 |
| Concentration | maxx/minn < 50 bps, avg < 20 bps |
| In-Sample phase | Mar 16 – Jun 10, 2026 (max 200 submissions) |
| OOS phase | May 15 – Jun 10, 2026 (max 20 submissions) |
| Final presentation | June 5, 2026 |

## Project Structure

```
.
├── core/
│   ├── datahub.py          # Data loading & 1-min → 15-min aggregation
│   ├── evaluator.py        # IC / IR / tvr / concentration metrics
│   ├── genalpha.py         # Evolutionary alpha generator (DSL + mutation)
│   ├── submission.py       # Submission packaging & gate checks
│   └── combiner.py         # Multi-factor ensemble combiner
├── frontend/               # Vite/React dashboard UI
├── factors/                # Saved factor formula files
├── submit/                 # Packaged submission artifacts (*.pq excluded from git)
├── outputs/                # Backtest outputs
├── leaderboard.py          # Local leaderboard tracker
├── evaluate_alpha.py       # CLI: evaluate a single alpha
├── factor_idea_generator.py # LLM-assisted factor ideation
├── formula_parser.py       # DSL → AST parser
├── compliance_guard.py     # Leakage & restriction checker
└── server.py               # FastAPI backend for the UI
```

## DSL Formula Language

Alphas are expressed in a proprietary DSL that maps to an AST evaluated over 15-min bar data.

- **Time-series**: `ts_mean`, `ts_std`, `ts_corr`, `ts_rank`, `ts_decay_linear`, `delta`, `delay`
- **Cross-sectional**: `rank`, `zscore`, `demean`, `scale`
- **Math**: `abs`, `log`, `sign`, `ifelse`, `+`, `-`, `*`, `/`

Available fields: `open`, `high`, `low`, `close`, `volume`, `vwap`, `amount`

## Quick Start

```bash
# 1. Activate environment
conda activate alphaclaw

# 2. Start backend + frontend together
./start.sh

# Or separately:
python server.py          # FastAPI backend on :8000
cd frontend && npm run dev  # Vite frontend on :3000
```

## Evaluate & Submit a Factor

```bash
# Evaluate a formula string
python evaluate_alpha.py --formula "rank(ts_mean(close/vwap, 20))" --name alpha_001

# Package for submission (checks all quality gates before writing .pq)
python -m core.submission
```

## Compliance Rules

- **No future data**: `resp` and `trading_restriction` fields are forbidden in factor formulas.
- **Coverage**: Alpha grid must cover every trading day in the evaluation window.
- **Bounds**: Output must be finite and bounded; `zscore` or `scale` recommended as final step.

## Metrics Reference

| Metric | Formula | Gate |
|--------|---------|------|
| IC | mean daily Pearson corr(alpha, resp) × 100 | > 0.6 |
| IR | IC.mean() / IC.std() × √252 | > 2.5 |
| tvr | mean daily turnover | < 400 |
| maxx | max single-stock weight (bps) | < 50 |
| Score | `(IC - 0.0005×tvr) × √IR × 100` | higher is better |
