"""
建模共享配置 — 路径、常量、样式。

所有子模块通过此文件获取统一的：
    - 路径常量 (DATA_DIR, MODEL_DIR, FIGURES_DIR)
    - 列分类 (数值列 / 分类列 / 二元列)
    - matplotlib 样式
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ── 路径常量 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── 默认数据文件 ──────────────────────────────────────────────────
DEFAULT_DATAFILE = "saas_customer_churn_cleaned.csv"

# ── 列分类 ────────────────────────────────────────────────────────
ID_COL = "CustomerID"
TARGET_COL = "Churn"

# 数值特征（需要标准化）
NUMERICAL_FEATURES = [
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SupportTickets",
    "LastLoginDays",
    "FeatureUsageCount",
    "ticket_sentiment",  # NLP 情感分析特征
]

# 二元分类特征（Yes/No → 1/0）
BINARY_FEATURES = [
    "SeniorCitizen",      # 0/1 already
    "Partner",
    "Dependents",
    "PaperlessBilling",
    "PhoneService",
]

# 多元分类特征（需要 OneHot 编码）
MULTI_CATEGORY_FEATURES = [
    "Gender",
    "Contract",
    "PaymentMethod",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "CompanySize",
    "Industry",
]

# 所有特征列（不含 ID 和目标）
ALL_FEATURE_COLS = NUMERICAL_FEATURES + BINARY_FEATURES + MULTI_CATEGORY_FEATURES

# ── 随机种子 ──────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.25

# ── matplotlib 样式 ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
})
sns.set_style("whitegrid")
