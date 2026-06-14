"""
数据预处理模块 — 特征工程流水线。

处理步骤：
    1. 加载清洗后的数据
    2. 删除 ID 列
    3. 二元特征 → 0/1 编码
    4. 多元分类特征 → OneHot 编码
    5. 数值特征 → StandardScaler 标准化
    6. 目标列 → 0/1 编码
    7. train/test split (stratified)
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BINARY_FEATURES,
    DATA_DIR,
    DEFAULT_DATAFILE,
    ID_COL,
    MODEL_DIR,
    MULTI_CATEGORY_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)

# 全局 Transformers（fit 后保存，predict 时复用）
_scaler: StandardScaler | None = None
_encoder: OneHotEncoder | None = None
_feature_names_out: list[str] = []


def load_data(filename: str | None = None) -> pd.DataFrame:
    """加载清洗后的数据。"""
    path = DATA_DIR / (filename or DEFAULT_DATAFILE)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    print(f"[Preprocess] Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def preprocess(
    df: pd.DataFrame,
    fit: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    完整的数据预处理流水线。

    Parameters
    ----------
    df : pd.DataFrame
        原始清洗后数据。
    fit : bool
        True = fit_transform (训练时)，False = transform (预测时)。

    Returns
    -------
    X : np.ndarray
        特征矩阵。
    y : np.ndarray
        目标向量 (0/1)。
    feature_names : list[str]
        特征列名列表。
    """
    global _scaler, _encoder, _feature_names_out

    df = df.copy()

    # ── Step 1: 删除 ID 列 ────────────────────────────────────────
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # ── Step 2: 目标编码 ──────────────────────────────────────────
    y = (df[TARGET_COL] == "Yes").astype(int).values
    df = df.drop(columns=[TARGET_COL])

    # ── Step 3: 二元特征 → 0/1 ────────────────────────────────────
    for col in BINARY_FEATURES:
        if col not in df.columns:
            continue
        if col == "SeniorCitizen":
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # ── Step 4: 分离数值和分类特征 ────────────────────────────────
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    cat_cols = [c for c in MULTI_CATEGORY_FEATURES if c in df.columns]

    X_num = df[num_cols].values.astype(np.float64)
    X_cat = df[cat_cols].astype(str).values

    # ── Step 5: OneHot 编码 ───────────────────────────────────────
    if fit:
        _encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat_encoded = _encoder.fit_transform(X_cat)
    else:
        if _encoder is None:
            raise RuntimeError("Encoder not fitted. Run preprocess(fit=True) first.")
        X_cat_encoded = _encoder.transform(X_cat)

    # ── Step 6: 标准化 ────────────────────────────────────────────
    if fit:
        _scaler = StandardScaler()
        X_num_scaled = _scaler.fit_transform(X_num)
    else:
        if _scaler is None:
            raise RuntimeError("Scaler not fitted. Run preprocess(fit=True) first.")
        X_num_scaled = _scaler.transform(X_num)

    # ── Step 7: 拼接 ──────────────────────────────────────────────
    X = np.hstack([X_num_scaled, X_cat_encoded])

    # 特征名
    num_names = num_cols.copy()
    cat_names = _encoder.get_feature_names_out(cat_cols).tolist()
    _feature_names_out = num_names + cat_names

    print(f"[Preprocess] Feature matrix: {X.shape[0]:,} x {X.shape[1]}")
    print(f"  Numerical: {len(num_names)}  |  OneHot: {len(cat_names)}  |  "
          f"Target: {y.shape[0]:,}")
    churn_rate = y.mean()
    print(f"  Churn rate: {churn_rate:.2%}")

    return X, y, _feature_names_out


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"[Preprocess] Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}  "
          f"|  Test size: {test_size:.0%}")
    return X_train, X_test, y_train, y_test


def save_transformers() -> None:
    """保存 scaler 和 encoder 到 models/ 目录。"""
    if _scaler:
        joblib.dump(_scaler, MODEL_DIR / "scaler.pkl")
        print(f"[Preprocess] Scaler saved to models/scaler.pkl")
    if _encoder:
        joblib.dump(_encoder, MODEL_DIR / "encoder.pkl")
        print(f"[Preprocess] Encoder saved to models/encoder.pkl")


def load_transformers() -> tuple[StandardScaler, OneHotEncoder]:
    """加载保存的 scaler 和 encoder。"""
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    return scaler, encoder


# ── CLI 自检 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(">>> Preprocessing Pipeline Self-Check")
    print("=" * 60)
    df = load_data()
    X, y, names = preprocess(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_transformers()
    print()
    print("Feature names (first 20):", names[:20])
    print("\n[OK] Preprocessing pipeline works correctly.")
