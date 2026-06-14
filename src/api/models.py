"""
Pydantic 数据模型 — 请求/响应 Schema。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """客户输入数据（用于预测）。"""
    gender: str = Field(..., example="Male")
    senior_citizen: int = Field(..., ge=0, le=1, example=0)
    partner: str = Field(..., example="Yes")
    dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, le=100, example=24)
    monthly_charges: float = Field(..., gt=0, example=70.50)
    total_charges: float = Field(..., ge=0, example=1692.00)
    contract: str = Field(..., example="Month-to-month")
    payment_method: str = Field(..., example="Electronic check")
    paperless_billing: str = Field(..., example="Yes")
    phone_service: str = Field(..., example="Yes")
    multiple_lines: str = Field(..., example="No")
    internet_service: str = Field(..., example="Fiber optic")
    online_security: str = Field(..., example="No")
    online_backup: str = Field(..., example="Yes")
    device_protection: str = Field(..., example="No")
    tech_support: str = Field(..., example="No")
    streaming_tv: str = Field(..., example="Yes")
    streaming_movies: str = Field(..., example="Yes")
    support_tickets: int = Field(..., ge=0, example=3)
    last_login_days: int = Field(..., ge=0, example=15)
    feature_usage_count: int = Field(..., ge=0, example=5)
    company_size: str = Field(..., example="Small (10-50)")
    industry: str = Field(..., example="Technology")
    ticket_sentiment: float = Field(..., ge=-1, le=1, example=-0.35)


class ShapExplanation(BaseModel):
    """SHAP 解释信息。"""
    feature: str
    value: float
    shap_value: float
    contribution: str = Field(description="'+' for churn risk, '-' for retention")


class PredictionResponse(BaseModel):
    """预测响应。"""
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: int = Field(..., description="0=Retained, 1=Churned")
    risk_level: str = Field(..., description="Low / Medium / High / Critical")
    model_used: str
    top_factors: list[dict] = Field(default_factory=list, description="Top 5 SHAP features")
    shap_explanations: list[ShapExplanation] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str
    models_loaded: list[str]
    features_count: int
    database_connected: bool
