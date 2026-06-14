"""
Transform — 数据转换层。

职责：
    1. 特征预处理（编码、标准化）
    2. 加载已训练模型
    3. 生成每条记录的流失预测概率
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    对原始数据执行特征工程 + 模型预测。

    处理步骤：
        1. 加载 scaler / encoder / 模型
        2. 预处理特征 → 生成特征矩阵
        3. 用三个模型分别预测流失概率

    Parameters
    ----------
    df : pd.DataFrame
        原始客户数据（含 CustomerID）。

    Returns
    -------
    pd.DataFrame
        预测结果表，包含 CustomerID + 各模型预测概率。
    """
    print("[Transform] Loading transformers & models ...")

    # ── 加载 transformers ────────────────────────────────────────
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")

    # ── 加载模型 ──────────────────────────────────────────────────
    models = {}
    model_names = {
        "logistic_regression": "lr",
        "random_forest": "rf",
        "xgboost": "xgb",
    }
    for filename, alias in model_names.items():
        path = MODEL_DIR / f"{filename}.pkl"
        if path.exists():
            models[alias] = joblib.load(path)
            print(f"  [OK] Loaded: {filename}")
        else:
            print(f"  [SKIP] Not found: {filename}")

    if not models:
        raise RuntimeError("No trained models found in models/. Run modeling first.")

    # ── 构建特征矩阵 ──────────────────────────────────────────────
    from src.modeling.preprocessing import BINARY_FEATURES, MULTI_CATEGORY_FEATURES, NUMERICAL_FEATURES

    df_feat = df.copy()

    # 保留 CustomerID
    customer_ids = df_feat["CustomerID"].values if "CustomerID" in df_feat.columns else None

    # 二元特征编码
    for col in BINARY_FEATURES:
        if col not in df_feat.columns:
            continue
        if col == "SeniorCitizen":
            df_feat[col] = df_feat[col].astype(int)
        else:
            df_feat[col] = df_feat[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    num_cols = [c for c in NUMERICAL_FEATURES if c in df_feat.columns]
    cat_cols = [c for c in MULTI_CATEGORY_FEATURES if c in df_feat.columns]

    X_num = df_feat[num_cols].values.astype(np.float64)
    X_cat = df_feat[cat_cols].astype(str).values

    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = encoder.transform(X_cat)
    X = np.hstack([X_num_scaled, X_cat_encoded])

    print(f"[Transform] Feature matrix: {X.shape[0]:,} x {X.shape[1]}")

    # ── 预测 ──────────────────────────────────────────────────────
    results = {"CustomerID": customer_ids if customer_ids is not None else np.arange(len(df))}

    for alias, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        pred = model.predict(X)
        results[f"{alias}_probability"] = np.round(proba, 4)
        results[f"{alias}_prediction"] = pred.astype(int)
        churn_pct = pred.mean() * 100
        print(f"  [{alias.upper()}] Predicted churn: {churn_pct:.1f}%")

    result_df = pd.DataFrame(results)

    # ── 合并原始字段用于 users 表 ─────────────────────────────────
    # 保留关键业务字段
    user_cols = [
        "CustomerID", "Gender", "SeniorCitizen", "Partner", "Dependents",
        "Tenure", "MonthlyCharges", "TotalCharges", "Contract",
        "PaymentMethod", "PaperlessBilling", "PhoneService",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "SupportTickets", "LastLoginDays", "FeatureUsageCount",
        "CompanySize", "Industry",
    ]
    available_cols = [c for c in user_cols if c in df.columns]

    result_df = result_df.merge(
        df[available_cols], on="CustomerID", how="left"
    )

    print(f"[Transform] Output: {result_df.shape[0]:,} rows x {result_df.shape[1]} cols")
    return result_df


if __name__ == "__main__":
    from .extract import extract
    df = extract()
    result = transform(df)
    print(result.head(3))
