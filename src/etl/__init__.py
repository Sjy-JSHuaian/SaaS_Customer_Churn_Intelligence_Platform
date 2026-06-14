"""
ETL 数据管道包 — SaaS 客户流失预测系统。

模块结构：
    config.py     数据库连接配置
    extract.py    数据提取层 (CSV → DataFrame)
    transform.py  数据转换层 (特征工程 + 模型预测)
    load.py       数据加载层 (DataFrame → MySQL/SQLite)

使用方式：
    # 一键运行完整 ETL
    python -m src.etl

    # 编程调用
    from src.etl import run_etl
    run_etl()
"""

from __future__ import annotations

from .config import DATABASE_URL, DB_DIR
from .extract import extract
from .load import create_tables, get_stats, load, print_summary, query_high_risk
from .transform import transform

__all__ = [
    "extract",
    "transform",
    "load",
    "create_tables",
    "get_stats",
    "print_summary",
    "query_high_risk",
    "run_etl",
    "DATABASE_URL",
    "DB_DIR",
]


def run_etl() -> dict:
    """
    一键执行完整 ETL 流水线。

    Extract → Transform → Load → Summary
    """
    print("=" * 60)
    print(">>> ETL Pipeline - Churn Intelligence")
    print(f">>> Target DB: {DATABASE_URL}")
    print("=" * 60)

    # 1. Extract
    print()
    df_raw = extract()

    # 2. Transform
    print()
    df_transformed = transform(df_raw)

    # 3. Load
    print()
    create_tables(drop_first=True)
    counts = load(df_transformed)

    # 4. Summary
    print_summary()

    # 5. High-risk customers
    print("\n>>> Top 10 High-Risk Customers (XGBoost):")
    high_risk = query_high_risk(10)
    print(high_risk.to_string(index=False))

    print("\n" + "=" * 60)
    print(">>> ETL Complete!")
    print(f"    Users: {counts['users']:,} rows")
    print(f"    Predictions: {counts['predictions']:,} rows")
    print(f"    Database: {DB_DIR / 'churn_intelligence.db'}")
    print("=" * 60)

    return counts


if __name__ == "__main__":
    run_etl()
