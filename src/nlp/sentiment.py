"""
NLP 情感分析模块 — 工单文本情绪评分。

流程：
    1. 基于客户特征生成模拟工单文本（ticket_text）
    2. 使用 TextBlob 对工单文本进行情感打分
    3. 输出 ticket_sentiment 特征（-1 ~ 1，负值 = 负面情绪）

工单文本生成逻辑：
    - 流失客户 → 更多抱怨关键词 → 偏负面
    - 高工单数 + 长登录间隔 → 偏负面
    - 长期留存客户 → 偏中性/正面
    - 混合服务问题、计费投诉、功能咨询等场景
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from textblob import TextBlob

# ── 工单文本模板库 ────────────────────────────────────────────────

# 负面情绪短语（流失倾向）
NEGATIVE_TEMPLATES = [
    "I am very frustrated with the service. {detail}",
    "This is unacceptable. {detail} I want to cancel my subscription.",
    "I've been experiencing {detail} for weeks. Very disappointing.",
    "Your customer support is terrible. {detail} No one is helping me.",
    "I regret signing up. {detail} The platform keeps crashing.",
    "The pricing is too high for what we get. {detail}",
    "We are considering switching to a competitor because {detail}",
    "The {feature} feature is broken. {detail} Fix it or I'm leaving.",
    "I have submitted multiple tickets about {detail} with no resolution.",
    "Extremely poor experience. {detail} I want a refund.",
]

# 中性情绪短语
NEUTRAL_TEMPLATES = [
    "I have a question about {detail}. Can someone help?",
    "We need assistance with {detail}. Please advise.",
    "The {feature} feature could be improved. {detail}",
    "I noticed {detail}. Is this expected behavior?",
    "Could you clarify the billing for {detail}?",
    "We would like to upgrade our plan. {detail}",
    "Please help me configure {feature}. {detail}",
    "I need to add more users to our account. {detail}",
]

# 正面情绪短语（留存客户）
POSITIVE_TEMPLATES = [
    "Great product! {detail} We love using it.",
    "The new {feature} update is fantastic. {detail}",
    "Our team is very satisfied with the service. {detail}",
    "Thank you for the quick response on {detail}. Much appreciated.",
    "We've been using this for {detail} and it's been excellent.",
    "The integration with {feature} works perfectly. {detail}",
    "Customer support was very helpful with {detail}. Thank you!",
    "I would recommend this to other teams. {detail}",
]

# 细节填充
DETAILS = [
    "the dashboard keeps showing wrong data",
    "billing charged us twice this month",
    "our API integration is failing",
    "the reports are not loading properly",
    "we can't add new team members",
    "the data export feature is missing",
    "login issues affecting the whole team",
    "performance has been very slow lately",
    "the mobile app keeps crashing",
    "email notifications are not working",
    "we need SSO integration for security",
    "the analytics don't match our internal numbers",
]

FEATURES = [
    "reporting", "analytics", "dashboard", "billing", "API",
    "user management", "notifications", "data export", "mobile",
    "security", "integration", "automation", "search",
]


def generate_ticket_text(row: pd.Series) -> str:
    """
    根据客户特征生成模拟工单文本。

    生成逻辑：
        - Churned 客户 → 70% 负面, 25% 中性, 5% 正面
        - 留存客户 + 高工单 → 40% 负面, 45% 中性, 15% 正面
        - 留存客户 + 低工单 → 10% 负面, 40% 中性, 50% 正面
        - LastLoginDays > 20 → 额外偏向负面
    """
    is_churned = (row.get("Churn", "No") == "Yes")
    tickets = row.get("SupportTickets", 0)
    login_days = row.get("LastLoginDays", 0)

    # 确定情绪倾向的概率权重
    if is_churned:
        p_neg, p_neu, p_pos = 0.70, 0.25, 0.05
    elif tickets >= 3 or login_days > 20:
        p_neg, p_neu, p_pos = 0.40, 0.45, 0.15
    elif tickets <= 1:
        p_neg, p_neu, p_pos = 0.10, 0.40, 0.50
    else:
        p_neg, p_neu, p_pos = 0.25, 0.50, 0.25

    # 随机选择模板
    rng = np.random.default_rng(hash(row.get("CustomerID", str(row.name))) % 2**31)
    choice = rng.choice(["neg", "neu", "pos"], p=[p_neg, p_neu, p_pos])

    if choice == "neg":
        template = rng.choice(NEGATIVE_TEMPLATES)
    elif choice == "neu":
        template = rng.choice(NEUTRAL_TEMPLATES)
    else:
        template = rng.choice(POSITIVE_TEMPLATES)

    detail = rng.choice(DETAILS)
    feature = rng.choice(FEATURES)

    return template.format(detail=detail, feature=feature)


def score_sentiment(text: str) -> float:
    """
    使用 TextBlob 对文本进行情感打分。

    Parameters
    ----------
    text : str
        待分析的文本。

    Returns
    -------
    float
        情感极性 (-1.0 = 极度负面, 0.0 = 中性, +1.0 = 极度正面)。
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


def add_ticket_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    为 DataFrame 添加工单文本情感分析列。

    生成两列：
        - ticket_text:      模拟工单文本
        - ticket_sentiment: 文本情感极性分 (-1 ~ 1)

    Parameters
    ----------
    df : pd.DataFrame
        客户数据（含 CustomerID, Churn, SupportTickets, LastLoginDays）。

    Returns
    -------
    pd.DataFrame
        添加了 ticket_text 和 ticket_sentiment 列的 DataFrame。
    """
    df = df.copy()
    np.random.seed(42)

    print("[NLP] Generating ticket text and sentiment scores ...")

    texts = []
    sentiments = []

    for _, row in df.iterrows():
        text = generate_ticket_text(row)
        texts.append(text)
        sentiments.append(score_sentiment(text))

    df["ticket_text"] = texts
    df["ticket_sentiment"] = sentiments

    # 统计
    neg_pct = (df["ticket_sentiment"] < -0.1).mean() * 100
    neu_pct = ((df["ticket_sentiment"] >= -0.1) & (df["ticket_sentiment"] <= 0.1)).mean() * 100
    pos_pct = (df["ticket_sentiment"] > 0.1).mean() * 100

    print(f"  [OK] Generated {len(df):,} ticket texts")
    print(f"  Sentiment distribution: "
          f"Negative={neg_pct:.1f}%  Neutral={neu_pct:.1f}%  Positive={pos_pct:.1f}%")
    print(f"  Mean polarity: {df['ticket_sentiment'].mean():.4f}  "
          f"(Churned: {df[df['Churn']=='Yes']['ticket_sentiment'].mean():.4f}, "
          f"Retained: {df[df['Churn']=='No']['ticket_sentiment'].mean():.4f})")

    return df


# ── CLI 自检 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    # 测试单条
    test_row = pd.Series({
        "CustomerID": "test-001",
        "Churn": "Yes",
        "SupportTickets": 4,
        "LastLoginDays": 25,
    })
    text = generate_ticket_text(test_row)
    score = score_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {score:.4f}")

    # 批量测试
    print("\n" + "=" * 50)
    path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "saas_customer_churn_cleaned.csv"
    df = pd.read_csv(path)
    df = add_ticket_sentiment(df.head(500))
    print(df[["CustomerID", "Churn", "SupportTickets", "ticket_sentiment", "ticket_text"]].head(5).to_string())
