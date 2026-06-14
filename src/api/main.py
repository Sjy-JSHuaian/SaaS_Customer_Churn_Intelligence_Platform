"""
FastAPI 主应用 — SaaS 客户流失预测服务。

接口：
    GET  /health            健康检查
    POST /predict           单条预测 + SHAP 解释
    POST /predict/batch     批量预测

启动：
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text

from .models import (
    CustomerInput,
    HealthResponse,
    PredictionResponse,
    ShapExplanation,
)

# ── 路径常量 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DB_DIR = PROJECT_ROOT / "database"
DB_URL = f"sqlite:///{DB_DIR / 'churn_intelligence.db'}"

# ── 全局变量 ──────────────────────────────────────────────────────
_scaler = None
_encoder = None
_model = None
_explainer = None
_feature_names: list[str] = []
_model_name: str = ""
_engine = None

# ── BINARY / CAT 列配置（与建模保持一致）───────────────────────────
BINARY_FEATURES = ["SeniorCitizen", "Partner", "Dependents", "PaperlessBilling", "PhoneService"]

MULTI_CATEGORY_FEATURES = [
    "Gender", "Contract", "PaymentMethod", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "CompanySize", "Industry",
]

NUMERICAL_FEATURES = [
    "Tenure", "MonthlyCharges", "TotalCharges",
    "SupportTickets", "LastLoginDays", "FeatureUsageCount",
    "ticket_sentiment",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期管理。"""
    global _scaler, _encoder, _model, _explainer, _feature_names, _model_name, _engine

    print("[API] Loading models and transformers ...")

    # XGBoost 作为默认模型（综合性能最好）
    model_path = MODEL_DIR / "xgboost.pkl"
    if not model_path.exists():
        # fallback
        model_path = MODEL_DIR / "logistic_regression.pkl"
    _model_name = model_path.stem

    _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    _encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    _model = joblib.load(model_path)

    # 构建特征名列表
    num_names = NUMERICAL_FEATURES.copy()
    cat_names = _encoder.get_feature_names_out(MULTI_CATEGORY_FEATURES).tolist()
    _feature_names = num_names + cat_names

    # 数据库引擎
    _engine = create_engine(DB_URL, echo=False)

    # SHAP Explainer (用 TreeExplainer 处理 XGBoost)
    try:
        _explainer = shap.TreeExplainer(_model)
        print("[API] SHAP TreeExplainer ready")
    except Exception:
        _explainer = shap.Explainer(_model, feature_names=_feature_names)
        print("[API] SHAP KernelExplainer ready (fallback)")

    print(f"[API] Model: {_model_name}  |  Features: {len(_feature_names)}  |  DB: connected")
    yield
    print("[API] Shutting down ...")


app = FastAPI(
    title="SaaS Customer Churn Intelligence API",
    description="ML-powered churn prediction with SHAP explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 辅助函数 ──────────────────────────────────────────────────────

def _preprocess_input(data: CustomerInput) -> np.ndarray:
    """将 CustomerInput 转为模型可用的特征矩阵。"""
    raw = data.model_dump()

    # Pydantic field name → CamelCase mapping
    FIELD_MAP = {
        "gender": "Gender", "senior_citizen": "SeniorCitizen",
        "partner": "Partner", "dependents": "Dependents",
        "tenure": "Tenure", "monthly_charges": "MonthlyCharges",
        "total_charges": "TotalCharges", "contract": "Contract",
        "payment_method": "PaymentMethod", "paperless_billing": "PaperlessBilling",
        "phone_service": "PhoneService", "multiple_lines": "MultipleLines",
        "internet_service": "InternetService", "online_security": "OnlineSecurity",
        "online_backup": "OnlineBackup", "device_protection": "DeviceProtection",
        "tech_support": "TechSupport", "streaming_tv": "StreamingTV",
        "streaming_movies": "StreamingMovies", "support_tickets": "SupportTickets",
        "last_login_days": "LastLoginDays", "feature_usage_count": "FeatureUsageCount",
        "company_size": "CompanySize", "industry": "Industry",
        "ticket_sentiment": "ticket_sentiment",
    }
    row = {FIELD_MAP.get(k, k): v for k, v in raw.items()}

    num_vals = [float(row[c]) for c in NUMERICAL_FEATURES]
    cat_vals = [str(row[c]) for c in MULTI_CATEGORY_FEATURES]

    # 二元特征编码
    bin_vals = []
    for c in BINARY_FEATURES:
        val = row.get(c, 0)
        if c == "SeniorCitizen":
            bin_vals.append(int(val))
        else:
            bin_vals.append(1 if val == "Yes" else 0)

    X_num = np.array(num_vals).reshape(1, -1)
    X_bin = np.array(bin_vals).reshape(1, -1)
    X_cat = np.array(cat_vals).reshape(1, -1)

    X_num_scaled = _scaler.transform(X_num)
    X_cat_encoded = _encoder.transform(X_cat)

    return np.hstack([X_num_scaled, X_cat_encoded])


def _compute_shap(X_row: np.ndarray) -> list[ShapExplanation]:
    """计算单行数据的 SHAP 值。"""
    shap_values = _explainer.shap_values(X_row)

    # SHAP 可能返回 list（多类）或 array
    if isinstance(shap_values, list):
        sv = shap_values[1]  # 正类
    else:
        sv = shap_values[0]

    explanations = []
    for i, (name, val, shap_val) in enumerate(
        zip(_feature_names, X_row[0], sv)
    ):
        if abs(shap_val) < 1e-6:
            continue
        explanations.append(ShapExplanation(
            feature=name,
            value=round(float(val), 4),
            shap_value=round(float(shap_val), 6),
            contribution="+" if shap_val > 0 else "-",
        ))

    # 按 |SHAP| 降序排列
    explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)
    return explanations


def _determine_risk(prob: float) -> str:
    """根据概率确定风险等级。"""
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"


def _save_to_db(customer_id: str, data: CustomerInput, prob: float, pred: int) -> None:
    """保存预测结果到数据库。"""
    with _engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO predictions (customer_id, model_name, probability, prediction)
            VALUES (:cid, :mn, :prob, :pred)
        """), {"cid": customer_id, "mn": _model_name, "prob": prob, "pred": pred})
        conn.commit()


# ── API 端点 ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查。"""
    db_ok = False
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        db_ok = False

    return HealthResponse(
        status="healthy",
        models_loaded=[_model_name],
        features_count=len(_feature_names),
        database_connected=db_ok,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerInput):
    """
    单条客户流失预测 + SHAP 解释。
    """
    try:
        customer_id = f"api-{uuid.uuid4().hex[:8]}"
        X = _preprocess_input(data)
        prob = float(_model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
        risk = _determine_risk(prob)
        shap_exps = _compute_shap(X)

        top_factors = [
            {
                "feature": e.feature,
                "shap_value": e.shap_value,
                "contribution": e.contribution,
            }
            for e in shap_exps[:5]
        ]

        # 保存到数据库
        try:
            _save_to_db(customer_id, data, prob, pred)
        except Exception:
            pass  # 数据库写入失败不影响主响应

        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=round(prob, 4),
            churn_prediction=pred,
            risk_level=risk,
            model_used=_model_name,
            top_factors=top_factors,
            shap_explanations=shap_exps[:20],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(records: list[CustomerInput]):
    """批量预测。"""
    results = []
    for data in records:
        customer_id = f"api-{uuid.uuid4().hex[:8]}"
        X = _preprocess_input(data)
        prob = float(_model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
        risk = _determine_risk(prob)

        results.append({
            "customer_id": customer_id,
            "churn_probability": round(prob, 4),
            "churn_prediction": pred,
            "risk_level": risk,
            "model_used": _model_name,
        })

    return {"results": results, "count": len(results)}


# ── CLI 启动入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
