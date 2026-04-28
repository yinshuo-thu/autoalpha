"""
验证平台评分公式，对比本地 submission_like 指标与官方回测指标。
运行：python research/score_formula_analysis/verify_score.py
"""
import math

# ─────────────────────────────────────────────
# 官方回测指标（平台返回）
# ─────────────────────────────────────────────
PLATFORM = {
    "alpha_001": {
        "IC": 2.1324, "IR": 5.5736, "tvr": 55.6497,
        "bl": 1.9949, "bs": 0.0, "nl": 4870.3822, "ns": 0.0,
        "nt": 4872.2685, "nd": 726.0,
        "maxx": 5.3013, "max": 4.1002, "minn": 0.0, "min": 0.0,
        "cover_all": 1, "score": 496.8581,
    },
    "alpha_007": {
        "IC": 1.6654, "IR": 24.2608, "tvr": 691.6246,
        "bl": 2.0, "bs": 0.0, "nl": 4883.638, "ns": 0.0,
        "nt": 4883.6398, "nd": 726.0,
        "maxx": 4.6357, "max": 4.0747, "minn": 0.0, "min": 0.0917,
        "cover_all": 1, "score": 0.0,
    },
    "alpha_006": {
        "IC": 1.7912, "IR": 28.2852, "tvr": 714.9574,
        "bl": 2.0, "bs": 0.0, "nl": 4883.638, "ns": 0.0,
        "nt": 4883.6398, "nd": 726.0,
        "maxx": 4.5109, "max": 4.07, "minn": 0.0, "min": 0.0917,
        "cover_all": 1, "score": 0.0,
    },
}

# 本地 submission_like 模式回测指标（来自 metadata.json）
LOCAL = {
    "alpha_001": {"IC": 2.3418, "IR": 6.2907, "tvr": 44.7293, "score": 581.7560},
    "alpha_006": {"IC": 1.5214, "IR": 29.0172, "tvr": 637.9845, "score": 0.0},
    "alpha_007": {"IC": 1.4366, "IR": 26.3690, "tvr": 618.6149, "score": 0.0},
}


def quality_gates(IC, IR, tvr, maxx, minn, max_w, min_w, cover_all):
    gates = {
        "cover_all": cover_all == 1,
        "IC > 0.6":   IC > 0.6,
        "IR > 2.5":   IR > 2.5,
        "tvr < 400":  tvr < 400,
        "maxx < 50":  maxx < 50,
        "|minn| < 50": abs(minn) < 50,
        "max < 20":   max_w < 20,
        "|min| < 20": abs(min_w) < 20,
    }
    return gates


def calc_score(IC, IR, tvr, gates_pass):
    if not gates_pass:
        return 0.0
    return max(0.0, IC - 0.0005 * tvr) * math.sqrt(IR) * 100


SEP = "=" * 60

print(f"\n{SEP}")
print("  平台评分公式验证：score = (IC - 0.0005 × tvr) × √IR × 100")
print(SEP)

for name, p in PLATFORM.items():
    IC, IR, tvr = p["IC"], p["IR"], p["tvr"]
    official_score = p["score"]
    gates = quality_gates(IC, IR, tvr, p["maxx"], p["minn"], p["max"], p["min"], p["cover_all"])
    all_pass = all(gates.values())

    calc = calc_score(IC, IR, tvr, all_pass)
    err = abs(calc - official_score)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  IC={IC}, IR={IR}, tvr={tvr}")
    print(f"  质量门：")
    for gate, ok in gates.items():
        mark = "✓" if ok else "✗ FAIL"
        print(f"    {gate:<18} {mark}")
    print(f"  all_pass = {all_pass}")
    print(f"  计算score = {calc:.4f}")
    print(f"  平台score = {official_score}")
    print(f"  误差     = {err:.6f}  {'✓ 吻合' if err < 0.02 else '⚠ 有偏差'}")

print(f"\n{SEP}")
print("  本地指标 vs 平台指标 对比")
print(SEP)

print(f"\n{'因子':<12}{'指标':<8}{'本地':>10}{'平台':>12}{'平台/本地':>12}")
print("─" * 56)
for name in ["alpha_001", "alpha_006", "alpha_007"]:
    loc = LOCAL[name]
    plat = PLATFORM[name]
    for key in ["IC", "IR", "tvr"]:
        ratio = plat[key] / loc[key] if loc[key] != 0 else float("inf")
        print(f"{name if key=='IC' else '':<12}{key:<8}{loc[key]:>10.4f}{plat[key]:>12.4f}{ratio:>12.4f}")

print(f"\n{SEP}")
print("  关键结论")
print(SEP)
print("""
1. 分数公式已验证（误差 < 0.02）:
   score = (IC - 0.0005 × tvr) × √IR × 100

2. 质量门条件（任一不满足即 score=0）:
   cover_all=1 | IC>0.6 | IR>2.5 | tvr<400
   maxx<50 | |minn|<50 | max<20 | |min|<20

3. alpha_006/007 均因 tvr >> 400 而归零。

4. 本地 submission_like 与平台的系统性偏差:
   - tvr: 平台 ≈ 本地 × 1.12~1.24  （本地低估换手率！）
   - IC:  可正可负偏，幅度 ±10~20%
   - IR:  偏差方向不稳定，幅度 ±10%

5. 风险提示: 若本地 tvr 在 320~400 之间，
   平台实际 tvr 可能超过 400，导致因子归零。
   建议以 tvr < 330 作为本地通过的保守阈值。
""")
