"""
EDA 探索性数据分析包 — SaaS 客户流失预测系统。

模块结构：
    config.py          共享配置（路径、样式、色板）
    churn_ratio.py     流失比例分析
    login_frequency.py 登录频率分析
    tenure_analysis.py 使用时长分析

使用方式：
    # 单独运行某个模块
    python -m src.eda.churn_ratio
    python -m src.eda.login_frequency
    python -m src.eda.tenure_analysis

    # 一键运行全部
    python -m src.eda

    # 编程调用
    from src.eda import churn_ratio, login_frequency, tenure_analysis
    import pandas as pd
    df = pd.read_csv("data/raw/saas_customer_churn_raw.csv")
    churn_ratio.run(df)
    login_frequency.run(df)
    tenure_analysis.run(df)
"""

from __future__ import annotations

import pandas as pd

from . import churn_ratio, login_frequency, tenure_analysis
from .config import (
    CHURN_COLORS,
    CHURN_PALETTE,
    FIGURES_DIR,
    load_data,
    save_fig,
)

__all__ = [
    "churn_ratio",
    "login_frequency",
    "tenure_analysis",
    "load_data",
    "save_fig",
    "CHURN_COLORS",
    "CHURN_PALETTE",
    "FIGURES_DIR",
    "run_all",
]


def run_all(df: pd.DataFrame | None = None) -> dict:
    """
    一键执行全部 EDA 分析并保存图表到 reports/figures/。

    Parameters
    ----------
    df : pd.DataFrame, optional
        数据。若为 None 则自动加载默认数据集。

    Returns
    -------
    dict
        包含各模块分析结果的字典。
    """
    if df is None:
        df = load_data()

    print("=" * 60)
    print(">>> SaaS Customer Churn - EDA Pipeline")
    print(f">>> Output: {FIGURES_DIR}")
    print("=" * 60)

    results = {}
    results["churn_ratio"] = churn_ratio.run(df)
    results["login_frequency"] = login_frequency.run(df)
    results["tenure"] = tenure_analysis.run(df)

    # 汇总
    print("\n" + "=" * 60)
    print(">>> EDA Complete!")
    print("=" * 60)
    png_count = len(list(FIGURES_DIR.glob("*.png")))
    print(f"  Total charts: {png_count} (PNG + SVG)")
    print(f"  Output directory: {FIGURES_DIR}")

    print("\n  [Key Findings]:")
    print(f"    1. Overall churn rate: {results['churn_ratio']['overall_churn_rate']:.2%}")
    print(f"    2. Churned avg login recency: "
          f"{results['login_frequency']['last_login_stats']['churned_mean']:.1f} days")
    print(f"    3. Retained avg login recency: "
          f"{results['login_frequency']['last_login_stats']['retained_mean']:.1f} days")
    print(f"    4. Churned avg tenure: "
          f"{results['tenure']['tenure_stats']['churned_mean']:.1f} months")
    print(f"    5. Retained avg tenure: "
          f"{results['tenure']['tenure_stats']['retained_mean']:.1f} months")

    return results


# ── CLI 入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all()
