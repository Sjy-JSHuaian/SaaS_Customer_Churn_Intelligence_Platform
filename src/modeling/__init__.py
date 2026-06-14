"""
建模包 — SaaS 客户流失预测系统。

模块结构：
    config.py              共享配置（路径、列分类、样式）
    preprocessing.py       特征工程（编码、标准化、train/test split）
    logistic_regression.py LogisticRegression 训练与评估

使用方式：
    # 一键运行
    python -m src.modeling

    # 单独模块
    python -m src.modeling.logistic_regression

    # 编程调用
    from src.modeling.logistic_regression import run
    run()
"""

from __future__ import annotations

from . import preprocessing
from .logistic_regression import evaluate, run, train
from .preprocessing import load_data, preprocess, split_data

__all__ = [
    "preprocessing",
    "load_data",
    "preprocess",
    "split_data",
    "train",
    "evaluate",
    "run",
]
