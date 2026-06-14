"""
Streamlit Dashboard — SaaS Customer Churn Intelligence Platform.
中英文双语，中文优先展示。

启动：
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# ── 国际化文本字典 ────────────────────────────────────────────────
T = {
    "zh": {
        "title": "📊 SaaS 客户流失智能平台",
        "subtitle": "机器学习驱动的客户流失预测 · NLP情感分析 · SHAP可解释性",
        "tab_overview": "📈 总览",
        "tab_high_risk": "🔍 高风险客户",
        "tab_shap": "🧠 SHAP 可解释性",
        "tab_data": "📋 原始数据",
        "total_customers": "总客户数",
        "predicted_churn": "预测流失 (XGB)",
        "avg_prob": "平均流失概率",
        "models_deployed": "已部署模型",
        "risk_dist": "流失风险分布",
        "risk_by_contract": "各合同类型流失率",
        "risk_by_tenure": "各使用时长流失率",
        "high_risk_title": "🔍 高风险客户（概率 ≥ {threshold:.0%}）",
        "high_risk_count": "高风险客户数",
        "high_risk_profile": "高风险客户画像",
        "avg_tenure": "平均使用时长",
        "avg_monthly": "平均月费",
        "avg_login_interval": "平均登录间隔",
        "avg_tickets": "平均工单数",
        "top_contract": "最多合同类型",
        "neg_sentiment": "负面情感占比",
        "shap_title": "🧠 SHAP 模型可解释性",
        "shap_desc": "SHAP 量化每个特征对流失预测的贡献。红色 = 增加流失风险，蓝色 = 降低流失风险。",
        "shap_feature_table": "流失驱动因素 Top 5",
        "raw_data_title": "📋 客户数据与预测结果",
        "download_csv": "下载 CSV",
        "filter_contract": "合同类型",
        "filter_all": "全部",
        "risk_threshold": "流失风险阈值",
        "select_model": "选择模型",
        "footer": "SaaS 客户流失智能平台 | FastAPI + SHAP + Streamlit | 模型: LR · RF · XGBoost",
        "months": "个月",
        "days": "天",
    },
    "en": {
        "title": "📊 SaaS Customer Churn Intelligence Platform",
        "subtitle": "ML-powered churn prediction · NLP sentiment · SHAP explainability",
        "tab_overview": "📈 Overview",
        "tab_high_risk": "🔍 High Risk Customers",
        "tab_shap": "🧠 SHAP Explainability",
        "tab_data": "📋 Raw Data",
        "total_customers": "Total Customers",
        "predicted_churn": "Predicted Churn (XGB)",
        "avg_prob": "Avg Churn Probability",
        "models_deployed": "Models Deployed",
        "risk_dist": "Churn Risk Distribution",
        "risk_by_contract": "Churn Rate by Contract",
        "risk_by_tenure": "Churn Rate by Tenure",
        "high_risk_title": "🔍 High Risk Customers (probability ≥ {threshold:.0%})",
        "high_risk_count": "High Risk Count",
        "high_risk_profile": "High Risk Profile",
        "avg_tenure": "Avg Tenure",
        "avg_monthly": "Avg Monthly Charges",
        "avg_login_interval": "Avg Login Interval",
        "avg_tickets": "Avg Support Tickets",
        "top_contract": "Top Contract",
        "neg_sentiment": "Negative Sentiment %",
        "shap_title": "🧠 SHAP Model Explainability",
        "shap_desc": "SHAP quantifies each feature's contribution. Red = increases churn risk, Blue = decreases risk.",
        "shap_feature_table": "Top 5 Churn Drivers",
        "raw_data_title": "📋 Customer Data & Predictions",
        "download_csv": "Download CSV",
        "filter_contract": "Contract Type",
        "filter_all": "All",
        "risk_threshold": "Churn Risk Threshold",
        "select_model": "Select Model",
        "footer": "SaaS Customer Churn Intelligence Platform | FastAPI + SHAP + Streamlit | Models: LR · RF · XGBoost",
        "months": "months",
        "days": "days",
    },
}


def t(key: str) -> str:
    """获取当前语言的文本。"""
    return T[st.session_state.lang][key]


# ── 页面配置 ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="SaaS 客户流失智能平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 语言初始化（中文优先）──────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "zh"  # 中文优先

# ── 路径 ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "database" / "churn_intelligence.db"
DB_URL = f"sqlite:///{DB_PATH}"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# ── 启动诊断 ──────────────────────────────────────────────────────
_startup_ok = True
_startup_msgs = []

if not DB_PATH.exists():
    _startup_ok = False
    _startup_msgs.append(f"⚠️ Database not found: {DB_PATH}")

if not FIGURES_DIR.exists():
    _startup_ok = False
    _startup_msgs.append(f"⚠️ Figures dir not found: {FIGURES_DIR}")
else:
    png_count = len(list(FIGURES_DIR.glob("*.png")))
    html_count = len(list(FIGURES_DIR.glob("*.html")))
    _startup_msgs.append(f"📁 Figures: {png_count} PNG + {html_count} HTML found")

engine = create_engine(DB_URL, echo=False)
try:
    with engine.connect() as _conn:
        _conn.execute(text("SELECT 1"))
    _startup_msgs.append("🗄️ Database: connected")
except Exception as _e:
    _startup_ok = False
    _startup_msgs.append(f"⚠️ Database error: {_e}")


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


# ── 加载数据 ──────────────────────────────────────────────────────
df = load_data()

# ── 侧边栏 ────────────────────────────────────────────────────────
with st.sidebar:
    # 语言切换
    lang_options = {"🇨🇳 中文": "zh", "🇬🇧 English": "en"}
    selected_label = st.selectbox(
        "语言 / Language",
        options=list(lang_options.keys()),
        index=0,
    )
    st.session_state.lang = lang_options[selected_label]

    # 启动状态
    with st.expander("🔧 系统状态 / System Status", expanded=not _startup_ok):
        for msg in _startup_msgs:
            st.caption(msg)
        if _startup_ok:
            st.success("✅ 系统就绪 / System Ready" if st.session_state.lang == "zh" else "✅ System Ready")
        else:
            st.error("❌ 存在异常 / Issues Detected")

    st.markdown("---")

    # 模型选择
    model_labels = {
        "zh": {"xgboost": "XGBoost", "logistic_regression": "逻辑回归", "random_forest": "随机森林"},
        "en": {"xgboost": "XGBoost", "logistic_regression": "Logistic Regression", "random_forest": "Random Forest"},
    }
    model_option = st.selectbox(
        t("select_model"),
        options=["xgboost", "logistic_regression", "random_forest"],
        format_func=lambda x: model_labels[st.session_state.lang][x],
    )

    prob_col_map = {
        "xgboost": "xgb_churn_prob",
        "logistic_regression": "lr_churn_prob",
        "random_forest": "rf_churn_prob",
    }
    prob_col = prob_col_map[model_option]

    # 风险阈值
    threshold = st.slider(t("risk_threshold"), 0.0, 1.0, 0.5, 0.05)

    # 合同筛选
    contracts = [t("filter_all")] + sorted(df["contract"].dropna().unique().tolist())
    selected_contract = st.selectbox(t("filter_contract"), contracts)

    st.markdown("---")

    filtered = df if selected_contract == t("filter_all") else df[df["contract"] == selected_contract]
    high_risk = filtered[filtered[prob_col] >= threshold]

    st.metric(t("total_customers"), f"{len(filtered):,}")
    st.metric(t("high_risk_count"), f"{len(high_risk):,}")
    st.metric(t("avg_prob"), f"{filtered[prob_col].mean():.2%}")

    st.markdown("---")
    st.caption(t("footer"))

# ═══════════════════════════════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════════════════════════════

st.title(t("title"))
st.markdown(t("subtitle"))

tab1, tab2, tab3, tab4 = st.tabs([
    t("tab_overview"), t("tab_high_risk"), t("tab_shap"), t("tab_data"),
])

# ═══════════════════════════════════════════════════════════════════
# Tab 1: Overview
# ═══════════════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(t("total_customers"), f"{len(df):,}")
    with col2:
        n_churn = len(df[df["xgb_pred"] == 1])
        st.metric(t("predicted_churn"), f"{n_churn:,}", f"{n_churn/len(df):.1%}")
    with col3:
        st.metric(t("avg_prob"), f"{df[prob_col].mean():.2%}")
    with col4:
        st.metric(t("models_deployed"), "3")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(t("risk_dist"))
        fig = px.histogram(
            df, x=prob_col, nbins=50,
            color_discrete_sequence=["#E74C3C"],
            labels={prob_col: "Churn Probability"},
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(t("risk_by_contract"))
        ct = df.groupby("contract")[prob_col].mean().reset_index()
        ct.columns = ["Contract", "Avg"]
        fig = px.bar(ct, x="Contract", y="Avg", color="Avg",
                     color_continuous_scale="Reds")
        fig.add_hline(y=df[prob_col].mean(), line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(t("risk_by_tenure"))
    tenure_bins = [0, 6, 12, 24, 36, 48, 60, 73]
    labels = ["0-6m", "6-12m", "12-24m", "24-36m", "36-48m", "48-60m", "60-72m"]
    df_plot = df.copy()
    df_plot["Bracket"] = pd.cut(df_plot["tenure"], bins=tenure_bins, labels=labels)
    ts = df_plot.groupby("Bracket", observed=False).agg(
        Avg=pd.NamedAgg(column=prob_col, aggfunc="mean"),
        Count=pd.NamedAgg(column="customer_id", aggfunc="count"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ts["Bracket"], y=ts["Avg"] * 100,
        marker_color=["#800026", "#BD0026", "#E31A1C", "#FC4E2A", "#FD8D3C", "#FEB24C", "#FED976"],
        text=[f"{v:.1f}%" for v in ts["Avg"] * 100],
        textposition="outside",
    ))
    fig.update_layout(xaxis_title="Tenure", yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# Tab 2: High Risk
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(t("high_risk_title").format(threshold=threshold))
    risk_df = filtered[filtered[prob_col] >= threshold].sort_values(prob_col, ascending=False)
    st.metric(t("high_risk_count"), f"{len(risk_df):,}")

    display_cols = [
        "customer_id", "tenure", "monthly_charges", "contract",
        "last_login_days", "support_tickets", prob_col,
    ]
    # ticket_sentiment 仅在存在时加入
    if "ticket_sentiment" in risk_df.columns:
        display_cols.insert(-1, "ticket_sentiment")

    available = [c for c in display_cols if c in risk_df.columns]
    style_cols = {prob_col: "{:.2%}"}
    if "ticket_sentiment" in available:
        style_cols["ticket_sentiment"] = "{:.3f}"

    st.dataframe(
        risk_df[available].head(50).style
        .background_gradient(subset=[prob_col], cmap="Reds")
        .format(style_cols),
        use_container_width=True, height=600,
    )

    # 高风险客户画像
    st.subheader(t("high_risk_profile"))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t("avg_tenure"), f"{risk_df['tenure'].mean():.1f} {t('months')}")
        st.metric(t("avg_monthly"), f"${risk_df['monthly_charges'].mean():.2f}")
    with col2:
        st.metric(t("avg_login_interval"), f"{risk_df['last_login_days'].mean():.1f} {t('days')}")
        st.metric(t("avg_tickets"), f"{risk_df['support_tickets'].mean():.1f}")
    with col3:
        top_ct = risk_df["contract"].value_counts().index[0]
        st.metric(t("top_contract"), top_ct)
        if "ticket_sentiment" in risk_df.columns:
            neg_pct = (risk_df["ticket_sentiment"] < 0).mean()
            st.metric(t("neg_sentiment"), f"{neg_pct:.1%}")

# ═══════════════════════════════════════════════════════════════════
# Tab 3: SHAP
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(t("shap_title"))
    st.markdown(t("shap_desc"))

    shap_files = [
        ("Summary Plot", "shap_summary.png"),
        ("Bar Plot (Importance)", "shap_bar.png"),
        ("Waterfall (Single Example)", "shap_waterfall.png"),
        ("Dependence Plot", "shap_dependence.png"),
    ]

    # SHAP 图片展示
    found_any = False
    for title, filename in shap_files:
        path = FIGURES_DIR / filename
        if path.exists():
            st.image(str(path), caption=title, use_container_width=True)
            found_any = True

    if not found_any:
        st.warning(
            "SHAP 图表未找到。请在本地运行 `python -m src.api.shap_viz` 生成图表后重新部署。"
            if st.session_state.lang == "zh"
            else "SHAP charts not found. Run `python -m src.api.shap_viz` locally and redeploy."
        )
        # 展示静态占位示例
        st.info(
            "📊 预期图表：Summary Plot · Bar Plot · Waterfall · Dependence Plot · Force Plot"
        )

    # Force Plot HTML
    force_path = FIGURES_DIR / "shap_force.html"
    if force_path.exists():
        with open(force_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=450, scrolling=True)
    else:
        st.info(
            "Force Plot 交互图未找到。运行 `python -m src.api.shap_viz` 生成。"
            if st.session_state.lang == "zh"
            else "Interactive Force Plot not found. Run `python -m src.api.shap_viz`."
        )

    st.subheader(t("shap_feature_table"))
    st.markdown("""
    | # | Feature / 特征 | Effect / 影响 |
    |---|----------------|---------------|
    | 1 | `ticket_sentiment` 工单情感 | 🔴 Negative → Churn |
    | 2 | `Contract_Month-to-month` 月付合同 | 🔴 Short contract → Churn |
    | 3 | `Tenure` 使用时长 | 🔵 Long tenure → Retain |
    | 4 | `Contract_Two year` 两年合同 | 🔵 Long contract → Retain |
    | 5 | `MonthlyCharges` 月费 | 🔴 High charges → Churn |
    """)

# ═══════════════════════════════════════════════════════════════════
# Tab 4: Raw Data
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(t("raw_data_title"))
    st.dataframe(
        filtered.sort_values(prob_col, ascending=False).head(200),
        use_container_width=True, height=600,
    )
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(t("download_csv"), csv, "churn_data.csv", "text/csv")

st.markdown("---")
st.caption(t("footer"))
