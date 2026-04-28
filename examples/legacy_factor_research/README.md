# Legacy Factor Research Examples

This directory keeps archived reference implementations that were useful during
v3 development and migration. They are intentionally separated from the main
runtime modules so the repository root stays focused on the active AutoAlpha v3
pipeline.

- `数据及因子/` contains legacy factor scripts, optimization experiments, and
  minute-index future-factor implementations.
- Raw data under nested `数据/` directories remains ignored by Git.
- Treat these files as references for idea mining, migration, and comparison;
  they are not imported by the primary `run.py` / `pipeline.py` flow.
