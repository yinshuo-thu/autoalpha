#!/usr/bin/env python3
"""
autoalpha_v2/run.py — Entry point

Usage:
    python autoalpha_v2/run.py               # 3 ideas, full trading history
    python autoalpha_v2/run.py --n 5         # 5 ideas
    python autoalpha_v2/run.py --days 120    # use last 120 trading days
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoalpha_v2.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="AutoAlpha: LLM → factor → parquet")
    parser.add_argument("--n",    type=int, default=3, help="Number of ideas to generate")
    parser.add_argument("--days", type=int, default=0, help="Number of recent trading days to evaluate; 0 means full history")
    args = parser.parse_args()

    print("=" * 60)
    print("  AutoAlpha Pipeline")
    print(f"  ideas={args.n}  eval_days={args.days}")
    print("=" * 60)

    results = run(n_ideas=args.n, eval_days_count=args.days)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for r in results:
        status = r.get("status", "?")
        formula = r.get("formula", "")[:60]
        if status == "ok":
            ic    = r.get("IC", 0)
            ir    = r.get("IR", 0)
            tvr   = r.get("tvr", 0)
            score = r.get("Score", 0)
            gates = r.get("PassGates", False)
            print(f"  ✓ {r['run_id']}")
            print(f"    formula : {formula}")
            print(f"    IC={ic:.3f}  IR={ir:.3f}  tvr={tvr:.1f}  "
                  f"pass={gates}  score={score:.3f}")
            if r.get("parquet_path"):
                print(f"    output  : {r['parquet_path']}")
        else:
            err = r.get("errors", r.get("error", "unknown"))
            print(f"  ✗ {r['run_id']}  [{status}]  {err}")

    passing = [r for r in results if r.get("PassGates")]
    print(f"\n  {len(passing)}/{len(results)} factors passed all quality gates.")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
