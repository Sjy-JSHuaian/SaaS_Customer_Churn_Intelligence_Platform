# 📊 SaaS Customer Churn Intelligence Platform

**ML-powered customer churn prediction with NLP sentiment analysis, SHAP explainability, and real-time dashboard.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.x-blueviolet)](https://shap.readthedocs.io)

---

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

### 2. Run EDA (Exploratory Data Analysis)

```bash
# Full EDA pipeline
python -m src.eda

# Individual modules
python -m src.eda.churn_ratio      # Churn ratio analysis
python -m src.eda.login_frequency   # Login behavior
python -m src.eda.tenure_analysis   # Customer tenure
```

Output: 11 charts saved to `reports/figures/`

### 3. Train Models

```bash
# Train all three models
python -m src.modeling.logistic_regression
python -m src.modeling.random_forest
python -m src.modeling.xgboost_model
```

### 4. Run ETL Pipeline

```bash
python -m src.etl
```

Writes 7,043 users and 21,129 predictions to `database/churn_intelligence.db`.

### 5. Start FastAPI Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

#### API Usage

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

Response includes `churn_probability`, `risk_level`, and `top_factors` with SHAP explanations.

### 6. Generate SHAP Visualizations

```bash
python -m src.api.shap_viz
```

Generates: `shap_summary.png`, `shap_bar.png`, `shap_waterfall.png`, `shap_force.html`, `shap_dependence.png`

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

*With NLP ticket_sentiment feature. All models use stratified 75/25 train/test split.*

## 🧠 NLP Sentiment Analysis

Generates realistic support ticket text and analyzes sentiment using **TextBlob**:

| Customer Type | Avg Sentiment |
|---------------|:-------------:|
| Churned | **-0.16** (negative) |
| Retained | **+0.07** (positive) |

The `ticket_sentiment` feature is the **#1 SHAP feature** (importance: 0.865), making it the most powerful predictor of churn.

## 📊 SHAP Top 10 Features

| Rank | Feature | SHAP Importance | Effect |
|:----:|---------|:---:|--------|
| 1 | `ticket_sentiment` | 0.865 | 🔴 Negative → churn |
| 2 | `Contract_Month-to-month` | 0.832 | 🔴 Short-term → churn |
| 3 | `Tenure` | 0.285 | 🔵 Longer → retain |
| 4 | `Contract_Two year` | 0.245 | 🔵 Long contract → retain |
| 5 | `MonthlyCharges` | 0.244 | 🔴 Higher → churn |
| 6 | `TotalCharges` | 0.216 | 🔵 Higher → retain |
| 7 | `LastLoginDays` | 0.205 | 🔴 Long absence → churn |
| 8 | `SupportTickets` | 0.160 | 🔴 More tickets → churn |
| 9 | `FeatureUsageCount` | 0.116 | 🔵 More usage → retain |
| 10 | `Contract_One year` | 0.069 | 🔵 Medium contract → retain |

## 🗄️ Database Schema

### `users` table (7,043 rows)
Customer demographics, subscriptions, services, and NLP sentiment.

### `predictions` table (21,129 rows)
Three model predictions per customer:

| Column | Description |
|--------|-------------|
| `customer_id` | Customer identifier |
| `model_name` | `logistic_regression` / `random_forest` / `xgboost` |
| `probability` | Churn probability (0–1) |
| `prediction` | Binary prediction (0=Retained, 1=Churned) |

## 🔧 Switching to MySQL

Edit `src/etl/config.py`:

```python
# SQLite (default)
DATABASE_URL = "sqlite:///database/churn_intelligence.db"

# MySQL
DATABASE_URL = "mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/churn_intelligence"
```

## 📋 Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
xgboost>=2.0
fastapi>=0.115
uvicorn>=0.30
streamlit>=1.28
plotly>=5.17
shap>=0.46
textblob>=0.17
pymysql>=1.1
sqlalchemy>=2.0
joblib>=1.4
pytest>=7.4
```

## 🤝 Contributing

1. Run `python -m src.eda` for data exploration
2. Run `python -m src.modeling.xgboost_model` for model training
3. Run `python -m src.etl` for ETL pipeline
4. Run API: `uvicorn src.api.main:app --reload`
5. Run Dashboard: `streamlit run src/dashboard/app.py`

---

**Built with FastAPI · XGBoost · SHAP · Streamlit · TextBlob**

🤖 Generated with [Claude Code](https://claude.com/claude-code)
