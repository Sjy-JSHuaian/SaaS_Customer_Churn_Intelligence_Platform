"""
SHAP 可视化模块 — 生成模型可解释性图表。

生成图表：
    - shap_summary.png      全局特征重要性（蜂群图）
    - shap_bar.png          全局特征重要性（柱状图）
    - shap_waterfall.png    单条预测瀑布图（示例）
    - shap_force.png        单条预测力图
    - shap_dependence.png   关键特征依赖图
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sqlalchemy import create_engine, text

matplotlib.use("Agg")

# ── 路径 ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DB_DIR = PROJECT_ROOT / "database"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── 全局 ──────────────────────────────────────────────────────────
_shap_values = None
_X_sample = None
_feature_names = None
_model = None


def generate_all() -> dict:
    """生成全部 SHAP 可视化图表。"""
    global _shap_values, _X_sample, _feature_names, _model

    print("=" * 60)
    print(">>> SHAP Explainability - Generating Visualizations")
    print("=" * 60)

    # 加载数据
    df = pd.read_csv(DATA_DIR / "saas_customer_churn_cleaned.csv")
    model = joblib.load(MODEL_DIR / "xgboost.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")

    _model = model

    # 特征工程（采样 500 条加速 SHAP）
    from src.modeling.preprocessing import BINARY_FEATURES, MULTI_CATEGORY_FEATURES, NUMERICAL_FEATURES

    df_sample = df.sample(min(500, len(df)), random_state=42).copy()

    for col in BINARY_FEATURES:
        if col in df_sample.columns:
            if col == "SeniorCitizen":
                df_sample[col] = df_sample[col].astype(int)
            else:
                df_sample[col] = df_sample[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    num_cols = [c for c in NUMERICAL_FEATURES if c in df_sample.columns]
    cat_cols = [c for c in MULTI_CATEGORY_FEATURES if c in df_sample.columns]

    X_num = df_sample[num_cols].values.astype(np.float64)
    X_cat = df_sample[cat_cols].astype(str).values

    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = encoder.transform(X_cat)
    X_sample = np.hstack([X_num_scaled, X_cat_encoded])

    num_names = num_cols.copy()
    cat_names = encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = num_names + cat_names

    _X_sample = X_sample
    _feature_names = feature_names

    # ── SHAP ─────────────────────────────────────────────────────
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    _shap_values = shap_values

    # 处理多类输出
    if isinstance(shap_values, list):
        sv = shap_values[1]
        exp_val = explainer.expected_value[1]
    else:
        sv = shap_values
        exp_val = explainer.expected_value

    sv = np.array(sv)
    print(f"[SHAP] Computed SHAP values: {sv.shape}")

    # ── 1. Summary Plot（蜂群图）────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      max_display=20, show=False)
    fig.tight_layout()
    path = FIGURES_DIR / "shap_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {path}")

    # ── 2. Bar Plot（柱状图）─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      plot_type="bar", max_display=20, show=False)
    fig.tight_layout()
    path = FIGURES_DIR / "shap_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {path}")

    # ── 3. Waterfall（第一行样本的瀑布图）────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv[0],
            base_values=exp_val,
            data=X_sample[0],
            feature_names=feature_names,
        ),
        max_display=10, show=False,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "shap_waterfall.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {path}")

    # ── 4. Force Plot（缓存为 HTML）──────────────────────────────
    force_html = shap.plots.force(
        exp_val, sv[0], X_sample[0],
        feature_names=feature_names,
        matplotlib=False,
    )
    shap.save_html(str(FIGURES_DIR / "shap_force.html"), force_html)
    print(f"  [OK] {FIGURES_DIR / 'shap_force.html'}")

    # ── 5. Dependence Plot（Top 1 特征）──────────────────────────
    top_idx = np.argmax(np.abs(sv).mean(axis=0))
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        top_idx, sv, X_sample,
        feature_names=feature_names,
        show=False, ax=ax,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "shap_dependence.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {path} (feature: {feature_names[top_idx]})")

    # ── 6. 特征重要性排序 ────────────────────────────────────────
    importance = np.abs(sv).mean(axis=0)
    top_indices = np.argsort(importance)[::-1][:15]

    print("\n  [SHAP] Top 15 Feature Importances:")
    for rank, idx in enumerate(top_indices, 1):
        name = feature_names[idx]
        imp = importance[idx]
        bar = "#" * max(1, int(imp / importance[top_indices[0]] * 40))
        print(f"    {rank:2d}. {name:<30s}  |SHAP|={imp:.6f}  {bar}")

    return {
        "top_features": [
            {"rank": i+1, "feature": feature_names[idx], "importance": float(importance[idx])}
            for i, idx in enumerate(top_indices[:10])
        ],
    }


if __name__ == "__main__":
    generate_all()
