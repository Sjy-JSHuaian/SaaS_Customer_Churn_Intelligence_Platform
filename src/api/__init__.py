"""
FastAPI 模型服务包 — SaaS 客户流失预测系统。

接口：
    GET  /health         健康检查
    POST /predict        单条预测 + SHAP 解释
    POST /predict/batch  批量预测

启动方式：
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
