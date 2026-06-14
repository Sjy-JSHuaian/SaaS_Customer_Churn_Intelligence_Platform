"""
流失比例分析模块 — EDA 子模块 (1/4)。

分析内容：
    - 整体流失率（饼图 + 柱状图）
    - 合同类型 vs 流失
    - 行业 vs 流失（Top 10）
    - 公司规模 vs 流失

输出图表：
    - churn_overall.png/svg       整体流失率
    - churn_by_contract.png/svg   合同类型 vs 流失
    - churn_by_industry.png/svg   行业 vs 流失
    - churn_by_company.png/svg    公司规模 vs 流失
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    CHURN_COLORS,
    CHURN_PALETTE,
    load_data,
    print_section_header,
    save_fig,
)


def run(df: pd.DataFrame | None = None) -> dict:
    """
    执行流失比例全面分析。

    Parameters
    ----------
    df : pd.DataFrame, optional
        数据。若为 None 则自动加载默认数据集。

    Returns
    -------
    dict
        包含 overall_churn_rate, churn_by_contract, churn_by_industry_top10,
        churn_by_company_size 等分析结果。
    """
    if df is None:
        df = load_data()

    print_section_header("Churn Ratio Analysis", "1/4")

    churn_counts = df["Churn"].value_counts()
    churn_rate = (df["Churn"] == "Yes").mean()

    print(f"  Overall Churn Rate: {churn_rate:.2%}")
    print(f"  Retained: {churn_counts.get('No', 0):,}    Churned: {churn_counts.get('Yes', 0):,}")

    _plot_overall(df, churn_counts)
    _plot_by_contract(df)
    _plot_by_industry(df, churn_rate)
    _plot_by_company_size(df)

    return {
        "overall_churn_rate": churn_rate,
        "churn_counts": churn_counts.to_dict(),
    }


# ── 内部绘图函数 ──────────────────────────────────────────────────

def _plot_overall(df: pd.DataFrame, churn_counts: pd.Series) -> None:
    """整体流失率：饼图 + 柱状图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 饼图
    wedges, texts, autotexts = axes[0].pie(
        churn_counts.values,
        labels=churn_counts.index.tolist(),
        colors=[CHURN_COLORS.get(k, "#999") for k in churn_counts.index],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0, 0.05),
        textprops={"fontsize": 13},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    axes[0].set_title("Overall Churn Rate (Pie)", fontweight="bold")

    # 柱状图
    bars = axes[1].bar(
        churn_counts.index.tolist(),
        churn_counts.values,
        color=[CHURN_COLORS.get(k, "#999") for k in churn_counts.index],
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, v in zip(bars, churn_counts.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{v:,}\n({v / len(df):.1%})",
            ha="center", fontsize=12, fontweight="bold",
        )
    axes[1].set_title("Overall Churn Count", fontweight="bold")
    axes[1].set_ylabel("Number of Customers")
    axes[1].set_ylim(0, churn_counts.max() * 1.18)

    fig.suptitle("SaaS Customer Churn - Overall Ratio", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "churn_overall")


def _plot_by_contract(df: pd.DataFrame) -> None:
    """合同类型 vs 流失：堆叠百分比 + 堆叠计数。"""
    ct = (
        df.groupby("Contract")["Churn"]
        .value_counts(normalize=False)
        .unstack(fill_value=0)
    )
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ct_pct.plot(
        kind="bar", stacked=True,
        color=CHURN_PALETTE,
        edgecolor="white", linewidth=1.0,
        ax=axes[0],
    )
    axes[0].set_title("Churn Rate by Contract Type", fontweight="bold")
    axes[0].set_xlabel("Contract")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].legend(title="Churn", loc="upper right")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="%.1f%%", fontsize=9, fontweight="bold")

    ct.plot(
        kind="bar", stacked=True,
        color=CHURN_PALETTE,
        edgecolor="white", linewidth=1.0,
        ax=axes[1],
    )
    axes[1].set_title("Customer Count by Contract Type", fontweight="bold")
    axes[1].set_xlabel("Contract")
    axes[1].set_ylabel("Number of Customers")
    axes[1].legend(title="Churn", loc="upper right")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    for c in axes[1].containers:
        axes[1].bar_label(c, fmt="%d", fontsize=8, fontweight="bold")

    fig.suptitle("Churn vs Contract Type", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "churn_by_contract")


def _plot_by_industry(df: pd.DataFrame, overall_rate: float) -> None:
    """行业 vs 流失（Top 10）：横向柱状图。"""
    ind = (
        df.groupby("Industry")["Churn"]
        .value_counts(normalize=False)
        .unstack(fill_value=0)
    )
    ind["Total"] = ind.sum(axis=1)
    ind["ChurnRate"] = (ind.get("Yes", 0) / ind["Total"]) * 100
    ind = ind.sort_values("Total", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("RdBu_r", len(ind))
    bars = ax.barh(
        ind.index.tolist()[::-1],
        ind["ChurnRate"].tolist()[::-1],
        color=colors[::-1],
        edgecolor="white", linewidth=1.0,
    )
    for bar, v in zip(bars, ind["ChurnRate"].tolist()[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
    avg_line = overall_rate * 100
    ax.axvline(avg_line, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Overall Avg: {avg_line:.1f}%")
    ax.set_title("Churn Rate by Industry (Top 10 by Customer Count)", fontweight="bold")
    ax.set_xlabel("Churn Rate (%)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "churn_by_industry")


def _plot_by_company_size(df: pd.DataFrame) -> None:
    """公司规模 vs 流失：堆叠百分比 + 堆叠计数。"""
    size_order = ["Micro (<10)", "Small (10-50)", "Medium (51-250)",
                  "Large (251-1000)", "Enterprise (1000+)"]
    cs = (
        df.groupby("CompanySize")["Churn"]
        .value_counts(normalize=False)
        .unstack(fill_value=0)
    )
    cs = cs.reindex([s for s in size_order if s in cs.index])
    cs_pct = cs.div(cs.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    cs_pct.plot(
        kind="bar", stacked=True,
        color=CHURN_PALETTE,
        edgecolor="white", linewidth=1.0,
        ax=axes[0],
    )
    axes[0].set_title("Churn Rate by Company Size", fontweight="bold")
    axes[0].set_xlabel("Company Size")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].legend(title="Churn", loc="upper right")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="%.1f%%", fontsize=8, fontweight="bold")

    cs.plot(
        kind="bar", stacked=True,
        color=CHURN_PALETTE,
        edgecolor="white", linewidth=1.0,
        ax=axes[1],
    )
    axes[1].set_title("Customer Count by Company Size", fontweight="bold")
    axes[1].set_xlabel("Company Size")
    axes[1].set_ylabel("Number of Customers")
    axes[1].legend(title="Churn", loc="upper right")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Churn vs Company Size", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "churn_by_company")


# ── CLI 入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
