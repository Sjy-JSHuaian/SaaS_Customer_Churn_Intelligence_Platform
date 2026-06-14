"""
使用时长分析模块 — EDA 子模块 (3/4)。

分析内容：
    - Tenure 分布（直方图 + KDE，按流失分层）
    - 使用时长 vs 流失（箱线图 + 小提琴图）
    - Tenure 分箱 vs 流失率
    - Tenure vs MonthlyCharges 联合分布散点图

输出图表：
    - tenure_distribution.png/svg        Tenure 分布
    - tenure_by_churn.png/svg            Tenure 按流失分组
    - tenure_bins_churn.png/svg          Tenure 分箱 vs 流失率
    - tenure_vs_monthly_charges.png/svg  Tenure vs 月费散点图
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
    执行使用时长全面分析。

    Parameters
    ----------
    df : pd.DataFrame, optional
        数据。若为 None 则自动加载默认数据集。

    Returns
    -------
    dict
        包含 tenure_stats, tenure_bin_stats 等分析结果。
    """
    if df is None:
        df = load_data()

    print_section_header("Tenure Analysis", "3/4")

    ten = df["Tenure"]
    churned = df[df["Churn"] == "Yes"]["Tenure"]
    retained = df[df["Churn"] == "No"]["Tenure"]

    print(f"  Tenure stats: mean={ten.mean():.1f} months  median={ten.median():.0f} months  "
          f"std={ten.std():.1f} months  range=[{ten.min()}, {ten.max()}]")
    print(f"  Churned  - mean: {churned.mean():.1f} months (median={churned.median():.0f})")
    print(f"  Retained - mean: {retained.mean():.1f} months (median={retained.median():.0f})")

    _plot_distribution(df)
    _plot_box_violin(df)
    bin_stats = _plot_tenure_bins(df)
    _plot_tenure_vs_charges(df)

    return {
        "tenure_stats": {
            "overall_mean": ten.mean(),
            "overall_median": ten.median(),
            "churned_mean": churned.mean(),
            "retained_mean": retained.mean(),
        },
        "tenure_bin_stats": bin_stats,
    }


# ── 内部绘图函数 ──────────────────────────────────────────────────

def _plot_distribution(df: pd.DataFrame) -> None:
    """Tenure 分布：直方图 + 分层 KDE。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 总体分布
    ax = axes[0]
    sns.histplot(df["Tenure"], bins=40, kde=True, color="#9B59B6",
                 edgecolor="white", linewidth=0.5, ax=ax)
    ax.axvline(df["Tenure"].mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {df['Tenure'].mean():.1f} months")
    ax.axvline(df["Tenure"].median(), color="orange", linestyle="--", linewidth=2,
               label=f"Median: {df['Tenure'].median():.0f} months")
    ax.set_title("Distribution of Customer Tenure", fontweight="bold")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Customer Count")
    ax.legend()

    # 按流失分层 KDE
    ax = axes[1]
    for label, color in [("No", CHURN_COLORS["No"]), ("Yes", CHURN_COLORS["Yes"])]:
        subset = df[df["Churn"] == label]["Tenure"]
        sns.kdeplot(subset, fill=True, alpha=0.35, color=color,
                    linewidth=2, label=f"Churn={label} (n={len(subset):,})", ax=ax)
    ax.set_title("Tenure Distribution by Churn", fontweight="bold")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Density")
    ax.legend()

    fig.suptitle("Customer Tenure Analysis", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "tenure_distribution")


def _plot_box_violin(df: pd.DataFrame) -> None:
    """箱线图 + 小提琴图：Tenure vs Churn。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(
        x="Churn", y="Tenure", data=df,
        palette=CHURN_PALETTE, width=0.4,
        ax=axes[0],
    )
    axes[0].set_title("Tenure vs Churn (Boxplot)", fontweight="bold")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Tenure (Months)")
    medians = df.groupby("Churn")["Tenure"].median()
    for i, (ch, med) in enumerate(medians.items()):
        axes[0].annotate(f"Median: {med:.0f}m", xy=(i, med),
                         fontsize=10, fontweight="bold",
                         ha="center", va="bottom",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    sns.violinplot(
        x="Churn", y="Tenure", data=df,
        palette=CHURN_PALETTE, inner="quartile",
        ax=axes[1],
    )
    axes[1].set_title("Tenure vs Churn (Violin)", fontweight="bold")
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Tenure (Months)")

    fig.suptitle("Tenure - Churned vs Retained", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "tenure_by_churn")


def _plot_tenure_bins(df: pd.DataFrame) -> dict:
    """Tenure 分箱 vs 流失率。"""
    bins = [0, 6, 12, 24, 36, 48, 60, 73]
    labels = ["0-6m", "6-12m", "12-24m", "24-36m", "36-48m", "48-60m", "60-72m"]
    df_bin = df.copy()
    df_bin["TenureBin"] = pd.cut(df["Tenure"], bins=bins, labels=labels, right=False)

    bin_stats = (
        df_bin.groupby("TenureBin", observed=False)
        .agg(
            CustomerCount=("Churn", "count"),
            ChurnCount=("Churn", lambda x: (x == "Yes").sum()),
            ChurnRate=("Churn", lambda x: (x == "Yes").mean() * 100),
            AvgMonthlyCharges=("MonthlyCharges", "mean"),
        )
        .reset_index()
    )

    print("\n  Tenure bin statistics:")
    for _, row in bin_stats.iterrows():
        bar = "#" * int(row["ChurnRate"] / 2)
        print(f"    {row['TenureBin']:<10s}  |  "
              f"Customers: {int(row['CustomerCount']):,}  |  "
              f"Churn Rate: {row['ChurnRate']:5.1f}%  {bar}  |  "
              f"Avg Monthly: ${row['AvgMonthlyCharges']:.2f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = sns.color_palette("viridis", len(bin_stats))
    bars = ax.bar(
        range(len(bin_stats)),
        bin_stats["ChurnRate"].values,
        color=colors,
        edgecolor="white", linewidth=1.2,
    )
    overall_rate = (df["Churn"] == "Yes").mean() * 100
    for i, (bar, row) in enumerate(zip(bars, bin_stats.itertuples())):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{row.ChurnRate:.1f}%\n(n={row.CustomerCount:,})",
            ha="center", fontsize=9, fontweight="bold",
        )
    ax.set_xticks(range(len(bin_stats)))
    ax.set_xticklabels(bin_stats["TenureBin"].tolist())
    ax.set_title("Churn Rate by Tenure Bracket", fontweight="bold")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Churn Rate (%)")
    ax.axhline(overall_rate, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Overall: {overall_rate:.1f}%")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "tenure_bins_churn")

    return bin_stats.to_dict()


def _plot_tenure_vs_charges(df: pd.DataFrame) -> None:
    """Tenure vs MonthlyCharges 散点图，按流失着色。"""
    fig, ax = plt.subplots(figsize=(9, 7))
    sample_n = min(3000, len(df))
    sns.scatterplot(
        x="Tenure", y="MonthlyCharges", hue="Churn",
        data=df.sample(sample_n, random_state=42),
        palette=CHURN_PALETTE, alpha=0.5, s=20,
        ax=ax,
    )
    ax.set_title("Tenure vs Monthly Charges (colored by Churn)", fontweight="bold")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Monthly Charges ($)")
    ax.legend(title="Churn")
    fig.tight_layout()
    save_fig(fig, "tenure_vs_monthly_charges")


# ── CLI 入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
