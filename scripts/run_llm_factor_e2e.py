"""
端到端：LLM 生成 1 条因子思路 -> quick_test 实算评测。

全量真实数据评测可能占用大量内存；若遇 OOM，可先设置环境变量：
  set AUTOALPHA_MOCK=1
（仍兼容 ALPHACLAW_MOCK=1）
使用小型合成数据跑通链路（指标为示意）。

用法（项目根目录）：
  python scripts/run_llm_factor_e2e.py
  set AUTOALPHA_MOCK=1 && python scripts/run_llm_factor_e2e.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("AUTOALPHA_USE_LLM", os.environ.get("ALPHACLAW_USE_LLM", "1"))


def main():
    from factor_idea_generator import generate_ideas_with_prompt
    from quick_test import quick_test
    from paths import LLM_MINING_JSONL

    prompt = (
        "假设：短期价格相对 VWAP 的偏离在日内存在均值回复。"
        "请给出一条仅使用 DSL 的因子公式，并保证可编译。"
    )
    print("[E2E] LLM 挖掘（1 条）...")
    ideas = generate_ideas_with_prompt(prompt, parents=[], num_ideas=1)
    if not ideas:
        print("[E2E] 无候选公式")
        return 1
    idea = ideas[0]
    formula = idea.get("formula") or ""
    rationale = idea.get("rationale") or idea.get("thought_process") or ""
    print(f"[E2E] formula: {formula}")
    print(f"[E2E] rationale: {rationale[:400]}...")

    name = "llm_e2e_001"
    print(f"\n[E2E] quick_test({name}) ...")
    result = quick_test(formula, name, hypothesis=rationale or None)

    print(json.dumps({k: result[k] for k in ("status", "IC", "rank_ic", "IR", "Turnover", "Score", "PassGates", "classification", "submission_ready_flag") if k in result}, indent=2, ensure_ascii=False, default=str))
    if result.get("metadata_path"):
        print(f"[E2E] metadata: {result['metadata_path']}")
    print(f"[E2E] mining log: {LLM_MINING_JSONL}")
    # success 表示公式可计算并完成评测；submission_ready 另需通关与 sanity
    return 0 if result.get("status") == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
