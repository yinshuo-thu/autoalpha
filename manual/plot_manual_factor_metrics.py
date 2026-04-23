from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MANUAL_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = MANUAL_ROOT / "artifacts"
REPORTS_DIR = MANUAL_ROOT / "reports"
MANIFEST_CSV = ARTIFACTS_DIR / "manual_factor_export_manifest_all.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(MANIFEST_CSV).sort_values("Score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["short_name"] = df["factor_name"] + " | " + df["family"]
    df["turnover_bucket"] = pd.cut(
        df["Turnover"],
        bins=[-np.inf, 2, 6, 10, np.inf],
        labels=["ultra_low", "low", "medium", "high"],
    ).astype(str)

    family = (
        df.groupby(["family", "family_label"], as_index=False)
        .agg(
            factor_count=("factor_name", "size"),
            best_score=("Score", "max"),
            mean_score=("Score", "mean"),
            best_ic=("IC", "max"),
            mean_turnover=("Turnover", "mean"),
            best_ir=("IR", "max"),
        )
        .sort_values(["best_score", "mean_score"], ascending=False)
        .reset_index(drop=True)
    )
    return df, family


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f5f1e8",
            "axes.facecolor": "#fffaf2",
            "axes.edgecolor": "#2f2a24",
            "axes.labelcolor": "#2f2a24",
            "xtick.color": "#2f2a24",
            "ytick.color": "#2f2a24",
            "text.color": "#2f2a24",
            "savefig.facecolor": "#f5f1e8",
            "savefig.bbox": "tight",
            "font.size": 12,
        }
    )


def annotate_barh(ax: plt.Axes, values: pd.Series, fmt: str = "{:.2f}") -> None:
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_width() + max(values) * 0.015,
            patch.get_y() + patch.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=10,
            color="#2f2a24",
        )


def make_dashboard(df: pd.DataFrame, family: pd.DataFrame) -> Path:
    top12 = df.head(12).copy()
    palette = sns.color_palette("crest", n_colors=len(top12))
    family_palette = dict(
        zip(family["family"], sns.color_palette("flare", n_colors=family["family"].nunique()))
    )

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.15], hspace=0.28, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(
        data=top12.sort_values("Score", ascending=True),
        x="Score",
        y="factor_name",
        hue="factor_name",
        palette=palette,
        legend=False,
        ax=ax1,
    )
    ax1.set_title("Top 12 Factors By Score", loc="left", fontsize=20, fontweight="bold")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("")
    annotate_barh(ax1, top12.sort_values("Score", ascending=True)["Score"], "{:.3f}")

    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(
        data=df,
        x="Turnover",
        y="Score",
        hue="family",
        size="IR",
        sizes=(80, 900),
        alpha=0.86,
        palette=family_palette,
        linewidth=0.8,
        edgecolor="#2f2a24",
        ax=ax2,
    )
    ax2.set_title("Score vs Turnover, Bubble Size = IR", loc="left", fontsize=20, fontweight="bold")
    ax2.set_xlabel("Turnover")
    ax2.set_ylabel("Score")
    label_df = df.head(8)
    for row in label_df.itertuples(index=False):
        ax2.text(row.Turnover + 0.2, row.Score + 0.05, row.factor_name, fontsize=9, alpha=0.9)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, title="Family / IR")

    ax3 = fig.add_subplot(gs[1, 0])
    heat = top12[["factor_name", "Score", "IC", "IR", "Turnover", "positive_ic_ratio", "maxx", "max_mean"]].copy()
    heat = heat.set_index("factor_name")
    # Convert turnover/concentration to "smaller is better" for easier visual comparison.
    heat["Turnover"] = -heat["Turnover"]
    heat["maxx"] = -heat["maxx"]
    heat["max_mean"] = -heat["max_mean"]
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min()).replace(0, 1)
    sns.heatmap(
        heat_norm,
        cmap=sns.color_palette("YlGnBu", as_cmap=True),
        linewidths=0.6,
        linecolor="#f5f1e8",
        cbar_kws={"label": "Normalized Quality"},
        ax=ax3,
    )
    ax3.set_title("Top 12 Factor Metric Heatmap", loc="left", fontsize=20, fontweight="bold")
    ax3.set_xlabel("")
    ax3.set_ylabel("")

    ax4 = fig.add_subplot(gs[1, 1])
    family_view = family.head(12).sort_values("best_score", ascending=True)
    sns.barplot(
        data=family_view,
        x="best_score",
        y="family_label",
        hue="family_label",
        palette=sns.color_palette("mako", n_colors=len(family_view)),
        legend=False,
        ax=ax4,
    )
    ax4.set_title("Best Score By Factor Family", loc="left", fontsize=20, fontweight="bold")
    ax4.set_xlabel("Best Score In Family")
    ax4.set_ylabel("")
    annotate_barh(ax4, family_view["best_score"], "{:.3f}")

    fig.suptitle("Manual Factor Comparison Dashboard", x=0.01, y=0.995, ha="left", fontsize=26, fontweight="bold")
    fig.text(
        0.01,
        0.968,
        "26 gate-passing factors exported under manual/submit. Strongest cluster: bar structure and range-location signals.",
        fontsize=13,
        ha="left",
    )

    out_path = REPORTS_DIR / "manual_factor_dashboard.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_distribution_panel(df: pd.DataFrame) -> Path:
    metrics = [
        ("Score", "#0f766e"),
        ("IC", "#1d4ed8"),
        ("IR", "#b45309"),
        ("Turnover", "#be123c"),
        ("positive_ic_ratio", "#7c3aed"),
        ("max_mean", "#374151"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    for ax, (col, color) in zip(axes, metrics):
        sns.histplot(df[col], kde=True, ax=ax, color=color, edgecolor="#fffaf2", alpha=0.85)
        mean_v = df[col].mean()
        median_v = df[col].median()
        ax.axvline(mean_v, color="#111827", linestyle="--", linewidth=1.6, label=f"mean {mean_v:.3f}")
        ax.axvline(median_v, color="#6b7280", linestyle=":", linewidth=1.6, label=f"median {median_v:.3f}")
        ax.set_title(col.replace("_", " ").title(), fontsize=16, fontweight="bold", loc="left")
        ax.legend(frameon=False, fontsize=10)
        ax.set_ylabel("Count")

    fig.suptitle("Distribution Of Key Metrics Across 26 Effective Factors", x=0.01, y=0.995, ha="left", fontsize=24, fontweight="bold")
    fig.text(
        0.01,
        0.965,
        "This panel shows whether performance is broad-based or dominated by a few outliers.",
        fontsize=13,
        ha="left",
    )
    out_path = REPORTS_DIR / "manual_factor_metric_distributions.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_rank_vs_metric_panel(df: pd.DataFrame) -> Path:
    top = df.sort_values("Score", ascending=False).copy()
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    metric_specs = [
        ("IC", "#0f766e", "IC by Score Rank"),
        ("IR", "#1d4ed8", "IR by Score Rank"),
        ("Turnover", "#b91c1c", "Turnover by Score Rank"),
    ]
    for ax, (metric, color, title) in zip(axes, metric_specs):
        ax.plot(top["rank"], top[metric], color=color, linewidth=2.5, marker="o", markersize=4)
        ax.set_title(title, loc="left", fontsize=17, fontweight="bold")
        ax.set_xlabel("Score Rank")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        if metric == "Turnover":
            best_low_turnover = top.nsmallest(3, "Turnover")
            for row in best_low_turnover.itertuples(index=False):
                ax.text(row.rank + 0.15, row.Turnover + 0.1, row.factor_name, fontsize=9)
        else:
            top_points = top.nlargest(3, metric)
            for row in top_points.itertuples(index=False):
                ax.text(row.rank + 0.15, getattr(row, metric), row.factor_name, fontsize=9)

    fig.suptitle("How Core Metrics Change Across The Factor Ranking", x=0.01, y=1.02, ha="left", fontsize=24, fontweight="bold")
    out_path = REPORTS_DIR / "manual_factor_rank_vs_metrics.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def write_summary(df: pd.DataFrame, family: pd.DataFrame, outputs: list[Path]) -> Path:
    top5 = df.head(5)
    low_turnover = df.nsmallest(5, "Turnover")[["factor_name", "family", "Turnover", "Score"]]
    best_family = family.head(8)[["family_label", "factor_count", "best_score", "mean_score", "mean_turnover"]]

    summary = f"""# Manual Factor Visual Summary

## Files
{"".join(f"- `{path.name}`\n" for path in outputs)}

## Quick Read
- Best factor by score: `{top5.iloc[0]['factor_name']}` ({top5.iloc[0]['family']}) with `Score={top5.iloc[0]['Score']:.3f}`.
- Strongest family cluster: `range-location / bar-structure` signals.
- Lowest-turnover winners come from `ema_spread`, which helps score by keeping turnover near zero.
- Performance is not a single-factor fluke: 26 factors passed gate, spanning 16 families.

## Top 5 By Score
{top5[['factor_name','family','Score','IC','IR','Turnover']].to_markdown(index=False)}

## Lowest Turnover Factors
{low_turnover.to_markdown(index=False)}

## Best Families
{best_family.to_markdown(index=False)}
"""

    out_path = REPORTS_DIR / "manual_factor_visual_summary.md"
    out_path.write_text(summary, encoding="utf-8")
    return out_path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    df, family = load_data()

    outputs = [
        make_dashboard(df, family),
        make_distribution_panel(df),
        make_rank_vs_metric_panel(df),
    ]
    summary_path = write_summary(df, family, outputs)

    print("[visual] generated:")
    for path in outputs + [summary_path]:
        print(path)


if __name__ == "__main__":
    main()
