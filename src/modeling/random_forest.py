"""
Random Forest 模型模块。

流程：
    1. 加载数据 → 预处理 → train/test split
    2. 训练 RandomForestClassifier (class_weight='balanced')
    3. 评估：Accuracy, Precision, Recall, F1, ROC-AUC
    4. 输出：confusion matrix, ROC curve, feature importance
    5. 保存模型到 models/
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

import matplotlib.pyplot as plt
import seaborn as sns


def train(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    训练 Random Forest 模型。

    使用 class_weight='balanced_subsample' 处理类别不平衡。
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"[RF] Training complete: {X_train.shape[0]:,} samples")
    print(f"[RF] Trees: {model.n_estimators}  |  Max depth: {model.max_depth}")
    return model


def evaluate(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict:
    """
    全面评估模型并生成可视化。
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }

    print("\n" + "=" * 50)
    print(">>> Random Forest - Evaluation")
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

    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_curve(y_test, y_proba, metrics["roc_auc"])

    if feature_names is not None:
        _plot_feature_importance(model, feature_names)

    return metrics


def _plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        annot_kws={"fontsize": 16, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title("Confusion Matrix - Random Forest", fontweight="bold", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    path = FIGURES_DIR / "rf_confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {path}")


def _plot_roc_curve(y_test: np.ndarray, y_proba: np.ndarray, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2C3E50", linewidth=2.5, label=f"RF (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Random Forest", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    path = FIGURES_DIR / "rf_roc_curve.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {path}")


def _plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    indices = np.argsort(model.feature_importances_)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_imp = model.feature_importances_[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(top_imp))
    bars = ax.barh(
        range(len(top_names))[::-1],
        top_imp[::-1],
        color=colors[::-1],
        edgecolor="white", linewidth=0.8,
    )
    ax.set_yticks(range(len(top_names))[::-1])
    ax.set_yticklabels([top_names[i] for i in range(len(top_names))][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importance - Random Forest",
                 fontweight="bold", fontsize=13)
    for bar, v in zip(bars, top_imp[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "rf_feature_importance.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {path}")


def save_model(model: RandomForestClassifier) -> None:
    import joblib
    path = MODEL_DIR / "random_forest.pkl"
    joblib.dump(model, path)
    print(f"[RF] Model saved to: {path}")


def run() -> dict:
    """一键执行完整的 Random Forest 训练 + 评估。"""
    from .preprocessing import load_data, preprocess, save_transformers, split_data

    print("=" * 60)
    print(">>> Random Forest - Full Pipeline")
    print("=" * 60)

    df = load_data()
    X, y, feature_names = preprocess(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print()
    model = train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test, feature_names)

    save_model(model)
    save_transformers()

    print("\n" + "=" * 60)
    print(">>> Pipeline Complete!")
    print("=" * 60)
    print(f"  Model: models/random_forest.pkl")
    print(f"  Scaler: models/scaler.pkl")
    print(f"  Encoder: models/encoder.pkl")

    return metrics


if __name__ == "__main__":
    run()
