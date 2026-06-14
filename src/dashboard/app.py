"""
Streamlit Dashboard — SaaS Customer Churn Intelligence Platform.

启动：
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# ── 页面配置 ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 路径 ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_URL = f"sqlite:///{PROJECT_ROOT / 'database' / 'churn_intelligence.db'}"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

engine = create_engine(DB_URL, echo=False)


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """从数据库加载用户和预测数据。"""
    query = """
        SELECT
            u.*,
            p_xgb.probability AS xgb_churn_prob,
            p_xgb.prediction  AS xgb_pred,
            p_lr.probability  AS lr_churn_prob,
            p_rf.probability  AS rf_churn_prob
        FROM users u
        LEFT JOIN predictions p_xgb
            ON u.customer_id = p_xgb.customer_id AND p_xgb.model_name = 'xgboost'
        LEFT JOIN predictions p_lr
            ON u.customer_id = p_lr.customer_id AND p_lr.model_name = 'logistic_regression'
        LEFT JOIN predictions p_rf
            ON u.customer_id = p_rf.customer_id AND p_rf.model_name = 'random_forest'
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)
    return df


@st.cache_data
def load_model_stats() -> dict:
    """从数据库加载模型统计。"""
    query = """
        SELECT model_name,
               COUNT(*) AS total,
               SUM(prediction) AS predicted_churn,
               ROUND(AVG(probability), 4) AS avg_prob
        FROM predictions
        GROUP BY model_name
    """
    with engine.connect() as conn:
        return pd.read_sql_query(text(query), conn).to_dict(orient="records")


# ── 加载数据 ──────────────────────────────────────────────────────
df = load_data()

# ── 侧边栏 ────────────────────────────────────────────────────────
st.sidebar.title("📊 Churn Intelligence")
st.sidebar.markdown("---")

# 模型选择
model_option = st.sidebar.selectbox(
    "Model",
    options=["xgboost", "logistic_regression", "random_forest"],
    format_func=lambda x: {"xgboost": "XGBoost", "logistic_regression": "Logistic Regression",
                           "random_forest": "Random Forest"}[x],
)

prob_col_map = {
    "xgboost": "xgb_churn_prob",
    "logistic_regression": "lr_churn_prob",
    "random_forest": "rf_churn_prob",
}
prob_col = prob_col_map[model_option]

# 风险阈值
threshold = st.sidebar.slider("Churn Risk Threshold", 0.0, 1.0, 0.5, 0.05)
high_risk = df[df[prob_col] >= threshold]

# 合同筛选
contracts = ["All"] + sorted(df["contract"].dropna().unique().tolist())
selected_contract = st.sidebar.selectbox("Contract Type", contracts)

filtered = df if selected_contract == "All" else df[df["contract"] == selected_contract]

st.sidebar.markdown("---")
st.sidebar.metric("Total Customers", f"{len(filtered):,}")
st.sidebar.metric("High Risk (≥{:.0%})".format(threshold), f"{len(high_risk):,}")
st.sidebar.metric("Avg Churn Probability", f"{filtered[prob_col].mean():.2%}")
st.sidebar.markdown("---")
st.sidebar.caption("Built with FastAPI + SHAP + Streamlit")

# ═══════════════════════════════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════════════════════════════

st.title("📊 SaaS Customer Churn Intelligence Platform")
st.markdown("ML-powered churn prediction & explainability dashboard")

# ── Tab 布局 ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Overview", "🔍 High Risk Customers", "🧠 SHAP Explainability", "📋 Raw Data"]
)

# ═══════════════════════════════════════════════════════════════════
# Tab 1: Overview
# ═══════════════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        high_risk_count = len(df[df["xgb_pred"] == 1])
        st.metric("Predicted Churn (XGB)", f"{high_risk_count:,}",
                  f"{high_risk_count/len(df):.1%}")
    with col3:
        avg_prob = df[prob_col].mean()
        st.metric("Avg Churn Probability", f"{avg_prob:.2%}")
    with col4:
        st.metric("Models Deployed", "3")

    st.markdown("---")

    # 风险分布
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Risk Distribution")
        fig = px.histogram(
            df, x=prob_col, nbins=50,
            color_discrete_sequence=["#E74C3C"],
            labels={prob_col: "Churn Probability"},
            title=f"Churn Probability Distribution ({model_option})",
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray",
                      annotation_text="Threshold=0.5")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Contract Type")
        contract_churn = df.groupby("contract")[prob_col].mean().reset_index()
        contract_churn.columns = ["Contract", "Avg_Churn_Prob"]
        fig = px.bar(
            contract_churn, x="Contract", y="Avg_Churn_Prob",
            color="Avg_Churn_Prob",
            color_continuous_scale="Reds",
            title="Average Churn Probability by Contract",
        )
        fig.add_hline(y=df[prob_col].mean(), line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    # Churn by Tenure Bracket
    st.subheader("Churn Rate by Tenure Bracket")
    tenure_bins = [0, 6, 12, 24, 36, 48, 60, 73]
    tenure_labels = ["0-6m", "6-12m", "12-24m", "24-36m", "36-48m", "48-60m", "60-72m"]
    df_plot = df.copy()
    df_plot["TenureBracket"] = pd.cut(df_plot["tenure"], bins=tenure_bins, labels=tenure_labels)
    tenure_stats = df_plot.groupby("TenureBracket", observed=False).agg(
        Avg_Churn=pd.NamedAgg(column=prob_col, aggfunc="mean"),
        Customer_Count=pd.NamedAgg(column="customer_id", aggfunc="count"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tenure_stats["TenureBracket"], y=tenure_stats["Avg_Churn"] * 100,
        name="Churn Rate (%)",
        marker_color=["#800026", "#BD0026", "#E31A1C", "#FC4E2A", "#FD8D3C", "#FEB24C", "#FED976"],
        text=[f"{v:.1f}%" for v in tenure_stats["Avg_Churn"] * 100],
        textposition="outside",
    ))
    fig.update_layout(title="Churn Rate by Customer Tenure", xaxis_title="Tenure", yaxis_title="Churn Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# Tab 2: High Risk Customers
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"🔍 High Risk Customers (Probability ≥ {threshold:.0%})")

    risk_df = filtered[filtered[prob_col] >= threshold].sort_values(prob_col, ascending=False)
    st.metric("High Risk Count", f"{len(risk_df):,}")

    display_cols = [
        "customer_id", "tenure", "monthly_charges", "contract",
        "last_login_days", "support_tickets", "ticket_sentiment", prob_col,
    ]
    available = [c for c in display_cols if c in risk_df.columns]
    st.dataframe(
        risk_df[available].head(50).style
        .background_gradient(subset=[prob_col], cmap="Reds")
        .format({prob_col: "{:.2%}", "ticket_sentiment": "{:.3f}"}),
        use_container_width=True, height=600,
    )

    # 高风险客户画像
    st.subheader("High Risk Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Tenure", f"{risk_df['tenure'].mean():.1f} months")
        st.metric("Avg Monthly Charges", f"${risk_df['monthly_charges'].mean():.2f}")
    with col2:
        st.metric("Avg Login Interval", f"{risk_df['last_login_days'].mean():.1f} days")
        st.metric("Avg Support Tickets", f"{risk_df['support_tickets'].mean():.1f}")
    with col3:
        top_contract = risk_df["contract"].value_counts().index[0]
        st.metric("Top Contract", top_contract)
        neg_sentiment = (risk_df["ticket_sentiment"] < 0).mean()
        st.metric("Negative Sentiment %", f"{neg_sentiment:.1%}")

# ═══════════════════════════════════════════════════════════════════
# Tab 3: SHAP Explainability
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🧠 SHAP Model Explainability")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) quantifies each feature's
    contribution to the churn prediction. Red = increases churn risk,
    Blue = decreases churn risk.
    """)

    col1, col2 = st.columns(2)

    # 显示 SHAP 图表
    shap_files = {
        "Summary Plot (Beeswarm)": "shap_summary.png",
        "Bar Plot (Importance)": "shap_bar.png",
        "Waterfall (Single Example)": "shap_waterfall.png",
        "Dependence Plot": "shap_dependence.png",
    }

    for title, filename in shap_files.items():
        path = FIGURES_DIR / filename
        if path.exists():
            st.image(str(path), caption=title, use_container_width=True)
        else:
            st.warning(f"SHAP chart not found: {filename}  (run `python -m src.api.shap_viz` first)")

    # SHAP Force Plot 交互
    st.subheader("Force Plot (Interactive)")
    force_path = FIGURES_DIR / "shap_force.html"
    if force_path.exists():
        with open(force_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=450, scrolling=True)
    else:
        st.info("Run `python -m src.api.shap_viz` to generate force plot.")

    # 特征重要性表
    st.subheader("Top Features Driving Churn")
    st.markdown("""
    | Rank | Feature | Effect |
    |------|---------|--------|
    | 1 | Contract_Month-to-month | 🔴 Increases churn risk |
    | 2 | Tenure | 🔵 Loyalty reduces risk |
    | 3 | MonthlyCharges | 🔴 High charges = higher risk |
    | 4 | ticket_sentiment | 🔴 Negative sentiment → churn |
    | 5 | LastLoginDays | 🔴 Long absence → at risk |
    """)

# ═══════════════════════════════════════════════════════════════════
# Tab 4: Raw Data
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Raw Customer Data & Predictions")

    st.dataframe(
        filtered.sort_values(prob_col, ascending=False).head(200),
        use_container_width=True, height=600,
    )

    # 导出按钮
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Data as CSV", csv,
        "churn_intelligence_data.csv", "text/csv",
    )


# ── 页脚 ──────────────────────────────────────────────────────────
st.markdown("---")
st.caption("SaaS Customer Churn Intelligence Platform | "
           "FastAPI + SHAP + Streamlit | "
           f"Models: LR · RF · XGBoost | "
           f"Data: {len(df):,} customers")
