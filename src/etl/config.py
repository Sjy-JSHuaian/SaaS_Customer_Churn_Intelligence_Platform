"""
ETL 共享配置 — 数据库连接、路径。

使用 SQLAlchemy，支持 SQLite（默认）和 MySQL 两种后端。
切换 MySQL 只需修改 DATABASE_URL。
"""

from __future__ import annotations

from pathlib import Path

# ── 路径常量 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
DB_DIR = PROJECT_ROOT / "database"

DB_DIR.mkdir(parents=True, exist_ok=True)

# ── 数据库配置 ────────────────────────────────────────────────────
# SQLite (默认，零配置)
DATABASE_URL = f"sqlite:///{DB_DIR / 'churn_intelligence.db'}"

# MySQL (切换到 MySQL 时取消注释下面这行)
# DATABASE_URL = "mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/churn_intelligence"

# ── 数据文件 ──────────────────────────────────────────────────────
DEFAULT_DATAFILE = "saas_customer_churn_cleaned.csv"
