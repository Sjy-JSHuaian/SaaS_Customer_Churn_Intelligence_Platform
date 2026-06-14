"""
登录频率分析模块 — EDA 子模块 (2/4)。

分析内容：
    - LastLoginDays 分布（直方图 + KDE，按流失分层）
    - 登录间隔 vs 流失（箱线图 + 小提琴图）
    - 登录间隔分箱 vs 流失率

输出图表：
    - login_distribution.png/svg   LastLoginDays 分布
    - login_by_churn.png/svg       LastLoginDays 按流失分组
    - login_bins_churn.png/svg     登录间隔分箱 vs 流失率
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
    执行登录频率全面分析。

    Parameters
    ----------
    df : pd.DataFrame, optional
        数据。若为 None 则自动加载默认数据集。

    Returns
    -------
    dict
        包含 last_login_stats, login_bin_stats 等分析结果。
    """
    if df is None:
        df = load_data()

    print_section_header("Login Frequency Analysis (LastLoginDays)", "2/4")

    lld = df["LastLoginDays"]
    churned = df[df["Churn"] == "Yes"]["LastLoginDays"]
    retained = df[df["Churn"] == "No"]["LastLoginDays"]

    print(f"  LastLoginDays stats: mean={lld.mean():.1f}  median={lld.median():.0f}  "
          f"std={lld.std():.1f}  min={lld.min()}  max={lld.max()}")
    print(f"  Churned  - mean={churned.mean():.1f}  median={churned.median():.0f}")
    print(f"  Retained - mean={retained.mean():.1f}  median={retained.median():.0f}")

    _plot_distribution(df)
    _plot_box_violin(df)
    bin_stats = _plot_login_bins(df)

    return {
        "last_login_stats": {
            "overall_mean": lld.mean(),
            "overall_median": lld.median(),
            "churned_mean": churned.mean(),
            "retained_mean": retained.mean(),
        },
        "login_bin_stats": bin_stats,
    }


# ── 内部绘图函数 ──────────────────────────────────────────────────

def _plot_distribution(df: pd.DataFrame) -> None:
    """LastLoginDays 分布：直方图 + 分层 KDE。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 总体分布
    ax = axes[0]
    sns.histplot(df["LastLoginDays"], bins=40, kde=True, color="#3498DB",
                 edgecolor="white", linewidth=0.5, ax=ax)
    ax.axvline(df["LastLoginDays"].mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {df['LastLoginDays'].mean():.1f} days")
    ax.axvline(df["LastLoginDays"].median(), color="orange", linestyle="--", linewidth=2,
               label=f"Median: {df['LastLoginDays'].median():.0f} days")
    ax.set_title("Distribution of Days Since Last Login", fontweight="bold")
    ax.set_xlabel("Days Since Last Login")
    ax.set_ylabel("Customer Count")
    ax.legend()

    # 按流失分层 KDE
    ax = axes[1]
    for label, color in [("No", CHURN_COLORS["No"]), ("Yes", CHURN_COLORS["Yes"])]:
        subset = df[df["Churn"] == label]["LastLoginDays"]
        sns.kdeplot(subset, fill=True, alpha=0.35, color=color,
                    linewidth=2, label=f"Churn={label} (n={len(subset):,})", ax=ax)
    ax.set_title("LastLoginDays Distribution by Churn", fontweight="bold")
    ax.set_xlabel("Days Since Last Login")
    ax.set_ylabel("Density")
    ax.legend()

    fig.suptitle("Login Recency Analysis", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "login_distribution")


def _plot_box_violin(df: pd.DataFrame) -> None:
    """箱线图 + 小提琴图：LastLoginDays vs Churn。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 箱线图
    sns.boxplot(
        x="Churn", y="LastLoginDays", data=df,
        palette=CHURN_PALETTE, width=0.4,
        ax=axes[0],
    )
    axes[0].set_title("LastLoginDays vs Churn (Boxplot)", fontweight="bold")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Days Since Last Login")
    medians = df.groupby("Churn")["LastLoginDays"].median()
    for i, (ch, med) in enumerate(medians.items()):
        axes[0].annotate(f"Median: {med:.0f}", xy=(i, med),
                         fontsize=10, fontweight="bold",
                         ha="center", va="bottom",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # 小提琴图
    sns.violinplot(
        x="Churn", y="LastLoginDays", data=df,
        palette=CHURN_PALETTE, inner="quartile",
        ax=axes[1],
    )
    axes[1].set_title("LastLoginDays vs Churn (Violin)", fontweight="bold")
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Days Since Last Login")

    fig.suptitle("Login Recency - Churned vs Retained", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "login_by_churn")


def _plot_login_bins(df: pd.DataFrame) -> dict:
    """登录间隔分箱 vs 流失率。"""
    bins = [0, 3, 7, 14, 30, 61]
    labels = ["0-3 days\n(Active)", "4-7 days\n(Recent)", "8-14 days\n(Moderate)",
              "15-30 days\n(Inactive)", "30+ days\n(At Risk)"]
    df_bin = df.copy()
    df_bin["LoginBin"] = pd.cut(df["LastLoginDays"], bins=bins, labels=labels,
                                right=False)

    bin_stats = (
        df_bin.groupby("LoginBin", observed=False)
        .agg(
            CustomerCount=("Churn", "count"),
            ChurnCount=("Churn", lambda x: (x == "Yes").sum()),
            ChurnRate=("Churn", lambda x: (x == "Yes").mean() * 100),
            AvgLastLogin=("LastLoginDays", "mean"),
        )
        .reset_index()
    )

    print("\n  Login bin statistics:")
    for _, row in bin_stats.iterrows():
        bar = "#" * int(row["ChurnRate"] / 2)
        print(f"    {row['LoginBin']:<22s}  |  "
              f"Customers: {int(row['CustomerCount']):,}  |  "
              f"Churn Rate: {row['ChurnRate']:5.1f}%  {bar}")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("RdBu_r", len(bin_stats))
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
            ha="center", fontsize=10, fontweight="bold",
        )
    ax.set_xticks(range(len(bin_stats)))
    ax.set_xticklabels(bin_stats["LoginBin"].tolist())
    ax.set_title("Churn Rate by Days Since Last Login", fontweight="bold")
    ax.set_xlabel("Days Since Last Login")
    ax.set_ylabel("Churn Rate (%)")
    ax.axhline(overall_rate, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Overall: {overall_rate:.1f}%")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "login_bins_churn")

    return bin_stats.to_dict()


# ── CLI 入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
