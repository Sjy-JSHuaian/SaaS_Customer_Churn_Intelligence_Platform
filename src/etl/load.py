"""
Load — 数据加载层。

将处理后的数据写入数据库：
    - users 表：客户画像
    - predictions 表：模型预测结果

使用 SQLAlchemy，SQLite / MySQL 通用。
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session

from .config import DATABASE_URL, DB_DIR

# ── SQLAlchemy Engine ─────────────────────────────────────────────

engine = create_engine(DATABASE_URL, echo=False)


class Base(DeclarativeBase):
    pass


# ── ORM 模型 ──────────────────────────────────────────────────────

class User(Base):
    """
    客户画像表。

    存储客户基本信息、订阅服务、使用行为等。
    """
    __tablename__ = "users"

    customer_id = Column(String(50), primary_key=True, comment="客户ID")
    gender = Column(String(10))
    senior_citizen = Column(Integer)
    partner = Column(String(5))
    dependents = Column(String(5))
    tenure = Column(Integer, comment="使用时长(月)")
    monthly_charges = Column(Float, comment="月费")
    total_charges = Column(Float, comment="总费用")
    contract = Column(String(30), comment="合同类型")
    payment_method = Column(String(50), comment="支付方式")
    paperless_billing = Column(String(5))
    phone_service = Column(String(5))
    internet_service = Column(String(30))
    online_security = Column(String(30))
    online_backup = Column(String(30))
    device_protection = Column(String(30))
    tech_support = Column(String(30))
    streaming_tv = Column(String(30))
    streaming_movies = Column(String(30))
    support_tickets = Column(Integer, comment="工单数")
    last_login_days = Column(Integer, comment="最近登录距今天数")
    feature_usage_count = Column(Integer, comment="功能使用数")
    company_size = Column(String(30))
    industry = Column(String(30))
    ticket_sentiment = Column(Float, comment="NLP 工单情感分数 (-1~1)")


class Prediction(Base):
    """
    流失预测结果表。

    存储三个模型的预测概率和预测标签。
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), index=True, comment="客户ID")
    model_name = Column(String(30), comment="模型名称: lr / rf / xgb")
    probability = Column(Float, comment="流失概率 (0~1)")
    prediction = Column(Integer, comment="预测标签 0=留存 1=流失")


# ── 建表和加载 ────────────────────────────────────────────────────


def create_tables(drop_first: bool = False) -> None:
    """
    创建 users 和 predictions 表。

    Parameters
    ----------
    drop_first : bool
        是否先删除旧表（默认 False，保留已有数据）。
    """
    if drop_first:
        Base.metadata.drop_all(engine)
        print("[Load] Dropped existing tables.")

    Base.metadata.create_all(engine)
    print(f"[Load] Tables created: users, predictions")
    print(f"[Load] Database: {DATABASE_URL}")


def load(df: pd.DataFrame, batch_size: int = 2000) -> dict:
    """
    将转换后的 DataFrame 写入数据库。

    Parameters
    ----------
    df : pd.DataFrame
        transform() 输出的完整数据。
    batch_size : int
        每批写入的行数。

    Returns
    -------
    dict
        {"users": n, "predictions": n}
    """
    print("[Load] Loading data into database ...")

    # ── DataFrame 列名 → DB 列名映射 ─────────────────────────────
    COLUMN_MAP = {
        "CustomerID": "customer_id",
        "Gender": "gender",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "Tenure": "tenure",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Contract": "contract",
        "PaymentMethod": "payment_method",
        "PaperlessBilling": "paperless_billing",
        "PhoneService": "phone_service",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "SupportTickets": "support_tickets",
        "LastLoginDays": "last_login_days",
        "FeatureUsageCount": "feature_usage_count",
        "CompanySize": "company_size",
        "Industry": "industry",
        "ticket_sentiment": "ticket_sentiment",  # NLP
    }

    # ── 写入 users 表 ────────────────────────────────────────────
    user_cols = [c.name for c in User.__table__.columns]
    # 重命名后只保留匹配的列
    user_df = df.rename(columns=COLUMN_MAP)
    user_df = user_df[[c for c in user_cols if c in user_df.columns]]
    user_df = user_df.drop_duplicates(subset=["customer_id"])

    user_df.to_sql(
        "users", engine,
        if_exists="replace",
        index=False,
        chunksize=batch_size,
    )
    print(f"  [OK] users: {len(user_df):,} rows")

    # ── 写入 predictions 表 ───────────────────────────────────────
    pred_rows = []
    model_aliases = {"lr": "logistic_regression", "rf": "random_forest", "xgb": "xgboost"}

    for alias, display_name in model_aliases.items():
        prob_col = f"{alias}_probability"
        pred_col = f"{alias}_prediction"
        if prob_col not in df.columns or pred_col not in df.columns:
            print(f"  [SKIP] {alias}: columns not found")
            continue

        for _, row in df.iterrows():
            pred_rows.append({
                "customer_id": row["CustomerID"],
                "model_name": display_name,
                "probability": float(row[prob_col]),
                "prediction": int(row[pred_col]),
            })

    pred_df = pd.DataFrame(pred_rows)

    pred_df.to_sql(
        "predictions", engine,
        if_exists="replace",
        index=False,
        chunksize=batch_size,
    )
    print(f"  [OK] predictions: {len(pred_df):,} rows ({len(model_aliases)} models)")

    counts = {"users": len(user_df), "predictions": len(pred_df)}
    return counts


def query_high_risk(top_n: int = 10) -> pd.DataFrame:
    """
    从数据库查询高风险客户（XGBoost 预测流失概率 > 0.5，按概率降序）。

    Parameters
    ----------
    top_n : int
        返回前 N 名。

    Returns
    -------
    pd.DataFrame
    """
    sql = text("""
        SELECT
            u.customer_id,
            u.tenure,
            u.monthly_charges,
            u.contract,
            u.last_login_days,
            p.model_name,
            ROUND(p.probability, 4) AS churn_probability
        FROM users u
        JOIN predictions p ON u.customer_id = p.customer_id
        WHERE p.model_name = 'xgboost'
          AND p.probability > 0.5
        ORDER BY p.probability DESC
        LIMIT :top_n
    """)
    with engine.connect() as conn:
        result = pd.read_sql_query(sql, conn, params={"top_n": top_n})
    return result


def get_stats() -> dict:
    """
    从数据库获取汇总统计。
    """
    stats = {}
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM users"))
        stats["total_users"] = result.scalar()

        result = conn.execute(text("SELECT COUNT(DISTINCT model_name) FROM predictions"))
        stats["total_models"] = result.scalar()

        result = conn.execute(text("""
            SELECT model_name,
                   COUNT(*) AS total,
                   SUM(prediction) AS predicted_churn,
                   ROUND(AVG(probability), 4) AS avg_prob
            FROM predictions
            GROUP BY model_name
        """))
        stats["by_model"] = [dict(row._mapping) for row in result]

    return stats


def print_summary() -> None:
    """打印数据库摘要。"""
    stats = get_stats()
    print("\n" + "=" * 50)
    print(">>> Database Summary")
    print("=" * 50)
    print(f"  Total users:    {stats['total_users']:,}")
    print(f"  Total models:   {stats['total_models']}")
    print(f"  Predictions by model:")
    for m in stats["by_model"]:
        rate = m["predicted_churn"] / m["total"] * 100
        print(f"    {m['model_name']:<22s}  "
              f"predicted_churn={m['predicted_churn']:,}  "
              f"rate={rate:.1f}%  "
              f"avg_prob={m['avg_prob']:.4f}")


if __name__ == "__main__":
    # 自检：创建空表
    create_tables(drop_first=True)
    print_summary()
