# 📊 SaaS Customer Churn Intelligence Platform / SaaS客户流失智能平台

**ML-powered customer churn prediction with NLP sentiment analysis, SHAP explainability, and real-time dashboard.**
**基于机器学习的SaaS客户流失预测系统，集成NLP情感分析、SHAP可解释性与实时仪表板。**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.x-blueviolet)](https://shap.readthedocs.io)

---

<!-- ============================================================ -->
<!-- Bilingual Toggle -->
<!-- ============================================================ -->
<details open>
<summary><b>🇬🇧 English</b></summary>

## 🎯 Overview

Predict SaaS customer churn with **3 ML models** (Logistic Regression, Random Forest, XGBoost), enriched with **NLP sentiment analysis** on support tickets, and explained with **SHAP** model interpretability. A **FastAPI** inference service and **Streamlit** dashboard provide real-time predictions and business insights.

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  data/raw/   │───▶│  preprocessing   │───▶│  data/processed/ │
│  CSV files   │    │  + NLP sentiment │    │  enhanced CSV    │
└──────────────┘    └────────┬─────────┘    └────────┬────────┘
                             │                       │
                             ▼                       ▼
                    ┌────────────────┐     ┌─────────────────┐
                    │  Model Training │     │   ETL Pipeline   │
                    │  LR · RF · XGB  │     │ Extract→Transform│
                    │  + SHAP         │     │ →Load to DB      │
                    └───────┬────────┘     └────────┬────────┘
                            │                       │
                            ▼                       ▼
                    ┌────────────────┐     ┌─────────────────┐
                    │  FastAPI       │     │  SQLite / MySQL  │
                    │  /predict      │◀────│  churn_intel.db  │
                    │  /health       │     │  users+predictions│
                    └───────┬────────┘     └────────┬────────┘
                            │                       │
                            ▼                       ▼
                    ┌─────────────────────────────────────────┐
                    │         Streamlit Dashboard              │
                    │  Overview · High Risk · SHAP · Raw Data  │
                    └─────────────────────────────────────────┘
```

## 📁 Project Structure

```
SaaS_Customer_Churn_Intelligence_Platform/
├── data/
│   ├── raw/                  # Raw CSV data
│   └── processed/            # Cleaned + NLP-enhanced data
├── models/                   # Trained models (pkl)
├── database/                 # SQLite database
├── reports/figures/          # EDA, SHAP, model evaluation charts
├── notebooks/                # Jupyter notebooks
├── src/
│   ├── data/                 # Data loading & preprocessing
│   │   ├── loader.py         # CSV/URL data loader
│   │   └── preprocess.py     # Cleaning pipeline
│   ├── eda/                  # Exploratory Data Analysis
│   │   ├── churn_ratio.py    # Churn ratio analysis
│   │   ├── login_frequency.py # Login behavior analysis
│   │   ├── tenure_analysis.py # Customer tenure analysis
│   │   └── config.py         # Shared config
│   ├── modeling/             # ML model training
│   │   ├── logistic_regression.py  # LR model
│   │   ├── random_forest.py        # RF model
│   │   ├── xgboost_model.py        # XGBoost model
│   │   ├── preprocessing.py        # Feature engineering
│   │   └── config.py               # Shared config
│   ├── nlp/                  # NLP sentiment analysis
│   │   └── sentiment.py      # Ticket text generation + TextBlob
│   ├── etl/                  # ETL pipeline
│   │   ├── extract.py        # CSV → DataFrame
│   │   ├── transform.py      # Feature engineering + predictions
│   │   └── load.py           # DataFrame → Database
│   ├── api/                  # FastAPI inference service
│   │   ├── main.py           # API endpoints (/predict, /health)
│   │   ├── models.py         # Pydantic schemas
│   │   └── shap_viz.py       # SHAP visualization generator
│   └── dashboard/            # Streamlit dashboard
│       └── app.py            # Interactive web dashboard
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run EDA

```bash
python -m src.eda                              # Full pipeline
python -m src.eda.churn_ratio                  # Churn ratio only
python -m src.eda.login_frequency               # Login behavior only
python -m src.eda.tenure_analysis               # Tenure analysis only
```

Output: 11 charts (PNG + SVG) in `reports/figures/`

### 3. Train Models

```bash
python -m src.modeling.logistic_regression
python -m src.modeling.random_forest
python -m src.modeling.xgboost_model
```

### 4. Run ETL Pipeline

```bash
python -m src.etl
```

Writes 7,043 users + 21,129 predictions to `database/churn_intelligence.db`.

### 5. Start FastAPI Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: `GET /health`
- **Predict**: `POST /predict` (with SHAP explanations)

#### API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "senior_citizen": 0, "partner": "No",
    "dependents": "No", "tenure": 3, "monthly_charges": 95.50,
    "total_charges": 286.50, "contract": "Month-to-month",
    "payment_method": "Electronic check", "paperless_billing": "Yes",
    "phone_service": "Yes", "multiple_lines": "No",
    "internet_service": "Fiber optic", "online_security": "No",
    "online_backup": "No", "device_protection": "No",
    "tech_support": "No", "streaming_tv": "Yes",
    "streaming_movies": "Yes", "support_tickets": 5,
    "last_login_days": 25, "feature_usage_count": 2,
    "company_size": "Small (10-50)", "industry": "Technology",
    "ticket_sentiment": -0.85
  }'
```

Response: `churn_probability`, `risk_level`, `top_factors` (SHAP explanations).

### 6. Generate SHAP Visualizations

```bash
python -m src.api.shap_viz
```

Output: `shap_summary.png`, `shap_bar.png`, `shap_waterfall.png`, `shap_force.html`, `shap_dependence.png`

### 7. Launch Streamlit Dashboard

```bash
streamlit run src/dashboard/app.py
```

Interactive dashboard with:
- 📈 **Overview**: Churn risk distribution, contract/tenure analysis
- 🔍 **High Risk Customers**: Filterable list with risk profiles
- 🧠 **SHAP Explainability**: Feature importance charts, force plots
- 📋 **Raw Data**: Exportable data table

## 🤖 Models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:---:|:-------:|
| Logistic Regression | 0.6843 | 0.4437 | 0.7516 | 0.5580 | 0.7733 |
| Random Forest | 0.7217 | 0.4833 | 0.7109 | 0.5754 | 0.7791 |
| **XGBoost** | **0.7388** | **0.5063** | **0.6060** | **0.5517** | **0.7766** |

*With NLP `ticket_sentiment` feature. 75/25 stratified split. Baseline (no NLP): LR 0.6315 / RF 0.6616 / XGB 0.6860 → **+5-8% improvement across all models**.*

## 🧠 NLP Sentiment

| Customer Type | Avg Sentiment |
|---------------|:-------------:|
| Churned | **-0.16** (negative) |
| Retained | **+0.07** (positive) |

`ticket_sentiment` is the **#1 SHAP feature** (importance: 0.865) — the strongest churn predictor.

## 📊 SHAP Top 10

| Rank | Feature | SHAP | Effect |
|:----:|---------|:---:|--------|
| 1 | `ticket_sentiment` | 0.865 | Negative → churn |
| 2 | `Contract_Month-to-month` | 0.832 | Short-term → churn |
| 3 | `Tenure` | 0.285 | Longer → retain |
| 4 | `Contract_Two year` | 0.245 | Long contract → retain |
| 5 | `MonthlyCharges` | 0.244 | Higher → churn |
| 6 | `TotalCharges` | 0.216 | Higher → retain |
| 7 | `LastLoginDays` | 0.205 | Long absence → churn |
| 8 | `SupportTickets` | 0.160 | More tickets → churn |
| 9 | `FeatureUsageCount` | 0.116 | More usage → retain |
| 10 | `Contract_One year` | 0.069 | Medium → retain |

## 🗄️ Database

### `users` — 7,043 rows
Customer demographics, subscriptions, services, NLP sentiment.

### `predictions` — 21,129 rows (3 models × 7,043 customers)

| Column | Description |
|--------|-------------|
| `customer_id` | Customer identifier |
| `model_name` | `logistic_regression` / `random_forest` / `xgboost` |
| `probability` | Churn probability (0–1) |
| `prediction` | 0=Retained, 1=Churned |

## 🔧 MySQL Setup

Edit `src/etl/config.py`:

```python
# SQLite (default)
DATABASE_URL = "sqlite:///database/churn_intelligence.db"

# MySQL
DATABASE_URL = "mysql+pymysql://root:PASSWORD@localhost:3306/churn_intelligence"
```

</details>

<!-- ============================================================ -->
<!-- Chinese -->
<!-- ============================================================ -->
<details>
<summary><b>🇨🇳 中文</b></summary>

## 🎯 项目概述

基于 **3种机器学习模型**（逻辑回归、随机森林、XGBoost）预测SaaS客户流失，集成 **NLP情感分析** 处理工单文本，通过 **SHAP** 提供模型可解释性。**FastAPI** 推理服务 + **Streamlit** 仪表板，提供实时预测和商业洞察。

## 🏗️ 系统架构

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  data/raw/   │───▶│   数据预处理      │───▶│  data/processed/ │
│  原始CSV     │    │  + NLP情感分析   │    │  增强数据集      │
└──────────────┘    └────────┬─────────┘    └────────┬────────┘
                             │                       │
                             ▼                       ▼
                    ┌────────────────┐     ┌─────────────────┐
                    │   模型训练      │     │   ETL 管道      │
                    │  LR·RF·XGB    │     │ 提取→转换→加载   │
                    │  + SHAP       │     │ →写入数据库      │
                    └───────┬────────┘     └────────┬────────┘
                            │                       │
                            ▼                       ▼
                    ┌────────────────┐     ┌─────────────────┐
                    │   FastAPI      │     │  SQLite/MySQL   │
                    │   /predict     │◀────│  churn_intel.db │
                    │   /health      │     │ 用户+预测数据    │
                    └───────┬────────┘     └────────┬────────┘
                            │                       │
                            ▼                       ▼
                    ┌─────────────────────────────────────────┐
                    │         Streamlit 仪表板                  │
                    │  概览 · 高风险 · SHAP · 原始数据           │
                    └─────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行 EDA 探索性分析

```bash
python -m src.eda                              # 一键全部分析
python -m src.eda.churn_ratio                  # 仅流失比例
python -m src.eda.login_frequency               # 仅登录频率
python -m src.eda.tenure_analysis               # 仅使用时长
```

输出 11 张图表到 `reports/figures/`

### 3. 训练模型

```bash
python -m src.modeling.logistic_regression
python -m src.modeling.random_forest
python -m src.modeling.xgboost_model
```

### 4. 运行 ETL 管道

```bash
python -m src.etl
```

将 7,043 条用户数据 + 21,129 条预测写入 `database/churn_intelligence.db`

### 5. 启动 FastAPI 服务

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger 文档**: http://localhost:8000/docs
- **健康检查**: `GET /health`
- **预测接口**: `POST /predict`（含 SHAP 解释）

### 6. 生成 SHAP 可视化

```bash
python -m src.api.shap_viz
```

### 7. 启动 Streamlit 仪表板

```bash
streamlit run src/dashboard/app.py
```

四大功能面板：
- 📈 **概览**：流失风险分布、合同/使用时长分析
- 🔍 **高风险客户**：可筛选高风险客户列表与画像
- 🧠 **SHAP 可解释性**：特征重要性图、力图、瀑布图
- 📋 **原始数据**：可导出的数据表格

## 🤖 模型表现

| 模型 | 准确率 | 精确率 | 召回率 | F1 | ROC-AUC |
|------|:---:|:---:|:---:|:---:|:---:|
| 逻辑回归 | 0.6843 | 0.4437 | 0.7516 | 0.5580 | 0.7733 |
| 随机森林 | 0.7217 | 0.4833 | 0.7109 | 0.5754 | 0.7791 |
| **XGBoost** | **0.7388** | **0.5063** | **0.6060** | **0.5517** | **0.7766** |

*含NLP `ticket_sentiment`特征。无NLP基线：LR 0.6315 / RF 0.6616 / XGB 0.6860 → **全面提升5-8个百分点**。*

## 🧠 NLP 情感分析

| 客户类型 | 平均情感分 |
|----------|:------:|
| 已流失 | **-0.16**（负面） |
| 留存 | **+0.07**（正面） |

`ticket_sentiment` 是 **SHAP 排名第一** 的特征（重要性 0.865）——最强的流失预测因子。

## 📊 SHAP 特征重要性 Top 10

| 排名 | 特征 | SHAP值 | 解读 |
|:--:|------|:---:|------|
| 1 | `ticket_sentiment` 工单情感 | 0.865 | 负面情绪 → 流失 |
| 2 | `Contract_Month-to-month` 月付合同 | 0.832 | 短期合同 → 流失 |
| 3 | `Tenure` 使用时长 | 0.285 | 长期客户 → 留存 |
| 4 | `Contract_Two year` 两年合同 | 0.245 | 长期合同 → 留存 |
| 5 | `MonthlyCharges` 月费 | 0.244 | 高月费 → 流失 |
| 6 | `TotalCharges` 总费用 | 0.216 | 高总费 → 留存 |
| 7 | `LastLoginDays` 最近登录 | 0.205 | 久未登录 → 流失 |
| 8 | `SupportTickets` 工单数 | 0.160 | 工单多 → 流失 |
| 9 | `FeatureUsageCount` 功能使用 | 0.116 | 多用 → 留存 |
| 10 | `Contract_One year` 一年合同 | 0.069 | 中期合同 → 留存 |

## 🗄️ 数据库

- `users` 表：7,043 行客户画像
- `predictions` 表：21,129 行预测记录（3模型 × 7,043 客户）

## 🔧 切换 MySQL

修改 `src/etl/config.py`：

```python
# SQLite（默认）
DATABASE_URL = "sqlite:///database/churn_intelligence.db"

# MySQL
DATABASE_URL = "mysql+pymysql://root:密码@localhost:3306/churn_intelligence"
```

</details>

---

## 📋 Requirements

```
numpy>=1.24   pandas>=2.0   scikit-learn>=1.3
matplotlib>=3.7   seaborn>=0.12   xgboost>=2.0
fastapi>=0.115   uvicorn>=0.30   streamlit>=1.28
plotly>=5.17   shap>=0.46   textblob>=0.17
pymysql>=1.1   sqlalchemy>=2.0   joblib>=1.4
```

---

**Built with FastAPI · XGBoost · SHAP · Streamlit · TextBlob**

🤖 Generated with [Claude Code](https://claude.com/claude-code)
