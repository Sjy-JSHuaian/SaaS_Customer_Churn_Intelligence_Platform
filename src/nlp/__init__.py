"""
NLP 自然语言处理包 — SaaS 客户流失预测系统。

模块结构：
    sentiment.py   工单文本情感分析（文本生成 + TextBlob 情感打分）

使用方式：
    from src.nlp import add_ticket_sentiment
    df = add_ticket_sentiment(df)  # 添加 ticket_sentiment 列
"""

from __future__ import annotations

from .sentiment import add_ticket_sentiment, generate_ticket_text, score_sentiment

__all__ = [
    "generate_ticket_text",
    "score_sentiment",
    "add_ticket_sentiment",
]
