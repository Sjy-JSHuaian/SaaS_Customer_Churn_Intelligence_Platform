"""
Logistic Regression 模型模块。

流程：
    1. 加载数据 → 预处理 → train/test split
    2. 训练 LogisticRegression (class_weight='balanced')
    3. 评估：Accuracy, Precision, Recall, F1, ROC-AUC
    4. 输出：confusion matrix, ROC curve
    5. 保存模型到 models/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import FIGURES_DIR, MODEL_DIR, RANDOM_STATE

# ── 绘图 ──────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import seaborn as sns


def train(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    训练 Logistic Regression 模型。

    使用 class_weight='balanced' 处理轻微类别不平衡 (~26.5% 正类)。
    """
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print(f"[LR] Training complete: {X_train.shape[0]:,} samples")
    print(f"[LR] Classes: {model.classes_}")
    print(f"[LR] Iterations: {model.n_iter_}")
    return model


def evaluate(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict:
    """
    全面评估模型并生成可视化。

    Returns
    -------
    dict
        metrics dict with keys: accuracy, precision, recall, f1, roc_auc.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── 基础指标 ──────────────────────────────────────────────────
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }

    print("\n" + "=" * 50)
    print(">>> Logistic Regression - Evaluation")
    print("=" * 50)
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")

    print("\n[Classification Report]")
    print(classification_report(
        y_test, y_pred,
        target_names=["Retained (0)", "Churned (1)"],
        digits=4,
    ))

    # ── 混淆矩阵 ──────────────────────────────────────────────────
    _plot_confusion_matrix(y_test, y_pred)

    # ── ROC 曲线 ──────────────────────────────────────────────────
    _plot_roc_curve(y_test, y_proba, metrics["roc_auc"])

    # ── 特征重要性 ────────────────────────────────────────────────
    if feature_names is not None:
        _plot_feature_importance(model, feature_names)

    return metrics


def _plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """绘制混淆矩阵。"""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        annot_kws={"fontsize": 16, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title("Confusion Matrix - Logistic Regression", fontweight="bold", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    path = FIGURES_DIR / "lr_confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {path}")


def _plot_roc_curve(y_test: np.ndarray, y_proba: np.ndarray, auc: float) -> None:
    """绘制 ROC 曲线。"""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2C3E50", linewidth=2.5, label=f"LR (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Logistic Regression", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    path = FIGURES_DIR / "lr_roc_curve.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {path}")


def _plot_feature_importance(
    model: LogisticRegression,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    """绘制 Top N 特征重要性（系数绝对值）。"""
    coef = model.coef_[0]
    importance = np.abs(coef)
    indices = np.argsort(importance)[::-1][:top_n]

    top_names = [feature_names[i] for i in indices]
    top_coef = [coef[i] for i in indices]
    top_importance = [importance[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#E74C3C" if c < 0 else "#2ECC71" for c in top_coef]
    bars = ax.barh(
        range(len(top_names))[::-1],
        top_importance[::-1],
        color=colors[::-1],
        edgecolor="white", linewidth=0.8,
    )
    ax.set_yticks(range(len(top_names))[::-1])
    ax.set_yticklabels([top_names[i] for i in range(len(top_names))][::-1], fontsize=9)
    ax.set_xlabel("|Coefficient| (absolute weight)")
    ax.set_title(f"Top {top_n} Feature Importance - Logistic Regression",
                 fontweight="bold", fontsize=13)
    ax.axvline(0, color="black", linewidth=0.5)

    # 在条形上方显示正/负标签
    for i, (bar, name, coef_val) in enumerate(zip(bars, top_names[::-1], top_coef[::-1])):
        sign = "(+)" if coef_val >= 0 else "(-)"
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{sign}", va="center", fontsize=8, fontweight="bold",
            color="#E74C3C" if coef_val < 0 else "#2ECC71",
        )

    fig.tight_layout()
    path = FIGURES_DIR / "lr_feature_importance.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {path}")


def save_model(model: LogisticRegression) -> None:
    """保存模型到 models/ 目录。"""
    import joblib
    path = MODEL_DIR / "logistic_regression.pkl"
    joblib.dump(model, path)
    print(f"[LR] Model saved to: {path}")


# ── 一键运行 ──────────────────────────────────────────────────────

def run() -> dict:
    """一键执行完整的 Logistic Regression 训练 + 评估。"""
    from .preprocessing import load_data, preprocess, save_transformers, split_data

    print("=" * 60)
    print(">>> Logistic Regression - Full Pipeline")
    print("=" * 60)

    # 1. 加载 & 预处理
    df = load_data()
    X, y, feature_names = preprocess(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 2. 训练
    print()
    model = train(X_train, y_train)

    # 3. 评估
    metrics = evaluate(model, X_test, y_test, feature_names)

    # 4. 保存
    save_model(model)
    save_transformers()

    # 5. 结果汇总
    print("\n" + "=" * 60)
    print(">>> Pipeline Complete!")
    print("=" * 60)
    print(f"  Model: models/logistic_regression.pkl")
    print(f"  Scaler: models/scaler.pkl")
    print(f"  Encoder: models/encoder.pkl")
    print(f"  Figures: {FIGURES_DIR}")

    return metrics


# ── CLI 入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
