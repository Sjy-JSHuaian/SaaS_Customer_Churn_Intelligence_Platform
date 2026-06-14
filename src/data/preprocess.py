"""
数据清洗模块 — SaaS 客户流失预测系统。

负责将 data/raw/ 中的原始数据清洗为标准化的干净数据，输出到
data/processed/。处理内容包括：

    1. 缺失值补全（数值→中位数，分类→众数）
    2. 重复行删除
    3. 异常值检测 & 盖帽（IQR 法，可配置）
    4. TotalCharges 一致性修复（TotalCharges ≈ MonthlyCharges × Tenure）
    5. 数据类型统一（数值→float/int，分类→category）

设计原则：
    - 幂等性：对已清洗的数据再次运行不会引入新问题。
    - 可追溯：每步清洗后打印日志，清楚记录变更。
    - 可配置：异常值阈值、策略均可参数化。
    - 无数据泄漏：不依赖任何 future information，适合部署前清洗。

使用方式：
    from src.data.preprocess import run_cleaning_pipeline
    result = run_cleaning_pipeline()  # 一键清洗

    # 或逐步执行：
    from src.data.preprocess import load_raw_data, clean_missing_values, ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ── 路径常量 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 确保目录存在
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── 列分类（用于针对性清洗策略）──────────────────────────────────
NUMERICAL_COLS = [
    "Tenure", "MonthlyCharges", "TotalCharges",
    "SupportTickets", "LastLoginDays", "FeatureUsageCount",
]

CATEGORICAL_COLS = [
    "Gender", "SeniorCitizen", "Partner", "Dependents",
    "Contract", "PaymentMethod", "PaperlessBilling",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "CompanySize", "Industry",
]

TARGET_COL = "Churn"
ID_COL = "CustomerID"

# IQR 异常值检测的默认倍数（1.5 = 标准；3.0 = 宽松）
DEFAULT_IQR_MULTIPLIER = 1.5

# TotalCharges 一致性检查的相对容差
TOTALCHARGES_RTOL = 0.05  # 5% 以内视为正常

# ── 加载原始数据 ──────────────────────────────────────────────────


def load_raw_data(
    filename: str = "saas_customer_churn_raw.csv",
) -> pd.DataFrame:
    """
    从 data/raw/ 加载原始数据集。

    Parameters
    ----------
    filename : str
        原始数据文件名。

    Returns
    -------
    pd.DataFrame
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"原始数据文件不存在: {path}")
    df = pd.read_csv(path)
    print(f"[clean] 加载原始数据: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    return df


# ── 缺失值处理 ────────────────────────────────────────────────────


def clean_missing_values(
    df: pd.DataFrame,
    *,
    num_strategy: str = "median",
    cat_strategy: str = "mode",
    num_cols: Optional[list[str]] = None,
    cat_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    补全缺失值。

    策略：
        - 数值列：默认中位数填充（抗异常值干扰）。
        - 分类列：默认众数填充。
        - 目标列 / ID 列：直接删除含缺失的行（不补全）。

    Parameters
    ----------
    df : pd.DataFrame
        包含缺失值的原始数据。
    num_strategy : str
        数值列填充策略：'median' | 'mean' | 'zero'。
    cat_strategy : str
        分类列填充策略：'mode' | 'missing_label'。
    num_cols : list[str], optional
        要处理的数值列，默认使用内置 NUMERICAL_COLS。
    cat_cols : list[str], optional
        要处理的分类列，默认使用内置 CATEGORICAL_COLS。

    Returns
    -------
    pd.DataFrame
        缺失值已补全的 DataFrame。
    """
    df = df.copy()
    num_cols = num_cols or [c for c in NUMERICAL_COLS if c in df.columns]
    cat_cols = cat_cols or [c for c in CATEGORICAL_COLS if c in df.columns]

    total_missing_before = df.isnull().sum().sum()
    if total_missing_before == 0:
        print("[clean] 未检测到缺失值，跳过。")
        return df

    changes = []

    # ── 数值列 ────────────────────────────────────────────────────
    for col in num_cols:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue

        if num_strategy == "median":
            fill_val = df[col].median()
        elif num_strategy == "mean":
            fill_val = df[col].mean()
        elif num_strategy == "zero":
            fill_val = 0
        else:
            raise ValueError(f"不支持的数值填充策略: {num_strategy}")

        df[col] = df[col].fillna(fill_val)
        changes.append(f"    {col}: {n_missing} 个缺失 → 用 {num_strategy}={fill_val:.2f} 填充")

    # ── 分类列 ────────────────────────────────────────────────────
    for col in cat_cols:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue

        if cat_strategy == "mode":
            fill_val = df[col].mode()
            fill_val = fill_val[0] if not fill_val.empty else "Unknown"
        elif cat_strategy == "missing_label":
            fill_val = "Missing"
        else:
            raise ValueError(f"不支持的分类填充策略: {cat_strategy}")

        df[col] = df[col].fillna(fill_val)
        changes.append(f"    {col}: {n_missing} 个缺失 → 用 {cat_strategy}={fill_val} 填充")

    # ── ID / 目标列 ──────────────────────────────────────────────
    for col in [ID_COL, TARGET_COL]:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                df = df.dropna(subset=[col])
                changes.append(f"    {col}: {n_missing} 个缺失 → 丢弃对应行")

    total_missing_after = df.isnull().sum().sum()
    if changes:
        print(f"[clean] 缺失值处理: {total_missing_before} → {total_missing_after}")
        for c in changes:
            print(c)

    return df


# ── 重复行处理 ────────────────────────────────────────────────────


def clean_duplicates(
    df: pd.DataFrame,
    *,
    subset: Optional[list[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """
    检测并删除重复行。

    默认会排除 ID 列后再检查重复（因为不同客户可能有相同属性）。
    若 subset 为空，则对所有非 ID 列去重。

    Parameters
    ----------
    df : pd.DataFrame
    subset : list[str], optional
        检查重复的列子集，默认排除 ID 列后的所有列。
    keep : str
        'first' | 'last' | False — 保留哪条重复记录。

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if subset is None:
        subset = [c for c in df.columns if c != ID_COL]

    dup_mask = df.duplicated(subset=subset, keep=False)
    n_dup = dup_mask.sum()

    if n_dup == 0:
        print("[clean] 未检测到重复行，跳过。")
        return df

    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    print(f"[clean] 删除重复行: {n_dup} → 0 (保留策略: {keep})")
    print(f"[clean] 清洗后: {df.shape[0]:,} 行")

    return df


# ── 异常值处理 ────────────────────────────────────────────────────


def detect_outliers_iqr(
    df: pd.DataFrame,
    col: str,
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> Tuple[float, float, pd.Series]:
    """
    使用 IQR 方法检测异常值。

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        要检测的数值列。
    multiplier : float
        IQR 倍数，默认 1.5。

    Returns
    -------
    (lower_bound, upper_bound, outlier_mask)
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    return lower, upper, mask


def clean_outliers(
    df: pd.DataFrame,
    *,
    num_cols: Optional[list[str]] = None,
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
    strategy: str = "cap",
) -> pd.DataFrame:
    """
    检测并处理异常值。

    策略：
        - 'cap'：盖帽法，将越界值裁剪到 IQR 边界（保留样本量）。
        - 'flag'：仅标记，新增 *_outlier 列（不修改原值，供后续分析）。

    Parameters
    ----------
    df : pd.DataFrame
    num_cols : list[str], optional
        要处理的数值列。默认排除 Tenure（业务上有 0 值的正常含义）。
    multiplier : float
        IQR 倍数。1.5 = 标准，3.0 = 宽松（仅标记极端异常）。
    strategy : str
        'cap' | 'flag'。

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # Tenure=0 是正常业务含义（新客户），不参与异常值检测
    # TotalCharges 是衍生指标 (=MonthlyCharges×Tenure)，由一致性检查负责
    if num_cols is None:
        num_cols = [
            c for c in NUMERICAL_COLS
            if c in df.columns and c not in ("Tenure", "TotalCharges")
        ]

    total_outliers = 0
    report_lines = []

    for col in num_cols:
        lower, upper, mask = detect_outliers_iqr(df, col, multiplier)
        n_out = mask.sum()
        if n_out == 0:
            continue

        total_outliers += n_out
        pct = n_out / len(df) * 100

        if strategy == "cap":
            n_below = (df[col] < lower).sum()
            n_above = (df[col] > upper).sum()
            df[col] = df[col].clip(lower, upper)
            report_lines.append(
                f"    {col:<22s} IQR=[{lower:8.2f}, {upper:8.2f}]  "
                f"盖帽 {n_out:4d} 个 ({pct:.2f}%)  "
                f"(低于下限: {n_below}, 高于上限: {n_above})"
            )

        elif strategy == "flag":
            df[f"{col}_outlier"] = mask.astype(int)
            report_lines.append(
                f"    {col:<22s} IQR=[{lower:8.2f}, {upper:8.2f}]  "
                f"标记 {n_out:4d} 个 ({pct:.2f}%)"
            )

    if report_lines:
        print(f"[clean] 异常值处理 (IQR × {multiplier}, 策略={strategy}): "
              f"共处理 {total_outliers} 个异常点")
        for line in report_lines:
            print(line)
    else:
        print("[clean] 未检测到 IQR 异常值，跳过。")

    return df


# ── TotalCharges 一致性修复 ──────────────────────────────────────


def clean_total_charges(
    df: pd.DataFrame,
    *,
    rtol: float = TOTALCHARGES_RTOL,
    fix: bool = True,
) -> pd.DataFrame:
    """
    检测并修复 TotalCharges 与 MonthlyCharges × Tenure 的不一致。

    真实数据集中常见问题：
        - Tenure = 0 但 TotalCharges > 0（新客户不应该有累计费用）
        - TotalCharges 与 MonthlyCharges × Tenure 偏差过大（录入错误）

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 Tenure, MonthlyCharges, TotalCharges 列。
    rtol : float
        相对容差。|实际 - 计算| / max(实际, 计算) > rtol 视为不一致。
    fix : bool
        若为 True，用 MonthlyCharges × Tenure 覆盖不一致的 TotalCharges。

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    required = {"Tenure", "MonthlyCharges", "TotalCharges"}
    missing = required - set(df.columns)
    if missing:
        print(f"[clean] TotalCharges 检查跳过: 缺少列 {missing}")
        return df

    # 确保数值类型
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    computed = df["MonthlyCharges"] * df["Tenure"]
    actual = df["TotalCharges"]

    # 相对偏差
    denominator = np.maximum(actual.abs(), computed.abs()).replace(0, np.nan)
    rel_diff = (actual - computed).abs() / denominator

    mask = (rel_diff > rtol) | (denominator.isna() & (actual != computed))
    n_inconsistent = mask.sum()

    if n_inconsistent == 0:
        print("[clean] TotalCharges 一致性检查通过，无异常。")
        return df

    print(f"[clean] TotalCharges 一致性异常: {n_inconsistent} 行 "
          f"({n_inconsistent / len(df):.2%})")

    # 按 Tenure 分段展示
    for desc, cond in [
        ("  Tenure=0 但 TotalCharges>0",
         (df["Tenure"] == 0) & (df["TotalCharges"] > 0)),
        ("  偏差 > {:.0%}".format(rtol),
         mask & ~((df["Tenure"] == 0) & (df["TotalCharges"] > 0))),
    ]:
        n = cond.sum()
        if n > 0:
            print(f"    {desc}: {n} 行")

    if fix:
        df.loc[mask, "TotalCharges"] = np.round(
            df.loc[mask, "MonthlyCharges"] * df.loc[mask, "Tenure"], 2
        )
        print("[clean] 已用 MonthlyCharges × Tenure 修复不一致的 TotalCharges。")

    return df


# ── 数据类型统一 ──────────────────────────────────────────────────


def clean_data_types(
    df: pd.DataFrame,
    *,
    num_cols: Optional[list[str]] = None,
    cat_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    统一数据类型：
        - 数值列强制转为 float（保留精度）或 int（如 Tenure, SeniorCitizen）。
        - 分类列转为 category 类型，节省内存并加速后续建模。
        - 目标列保留字符串。

    Parameters
    ----------
    df : pd.DataFrame
    num_cols : list[str], optional
    cat_cols : list[str], optional

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    num_cols = num_cols or [c for c in NUMERICAL_COLS if c in df.columns]
    cat_cols = cat_cols or [c for c in CATEGORICAL_COLS if c in df.columns]

    # 整数型数值列
    int_like = {"Tenure", "SeniorCitizen", "SupportTickets", "LastLoginDays",
                "FeatureUsageCount"}

    for col in num_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col in int_like:
            df[col] = df[col].fillna(0).astype(int)
        else:
            df[col] = df[col].astype(float)

    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype("category")

    print(f"[clean] 数据类型统一: {len(num_cols)} 个数值列, "
          f"{len(cat_cols)} 个分类列")
    return df


# === 完整清洗流水线 ===


def run_cleaning_pipeline(
    input_filename: str = "saas_customer_churn_raw.csv",
    output_filename: str = "saas_customer_churn_cleaned.csv",
    *,
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
    outlier_strategy: str = "cap",
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """
    一键执行完整数据清洗流水线。

    步骤:
        1. 加载 data/raw/ 原始数据
        2. 缺失值补全
        3. 重复行删除
        4. TotalCharges 一致性修复
        5. 异常值盖帽
        6. 数据类型统一
        7. 保存到 data/processed/

    Parameters
    ----------
    input_filename : str
        data/raw/ 下的原始文件名。
    output_filename : str
        输出到 data/processed/ 的文件名。
    iqr_multiplier : float
        IQR 异常值检测系数，默认 1.5。
    outlier_strategy : str
        'cap' 盖帽 | 'flag' 标记。
    save : bool
        是否保存清洗结果。
    verbose : bool
        是否打印详细步骤。

    Returns
    -------
    dict
        包含 df_clean, report, output_path 等信息。
    """
    report = {
        "input_file": input_filename,
        "rows_before": 0,
        "cols_before": 0,
        "rows_after": 0,
        "cols_after": 0,
        "steps": [],
    }

    # ── Step 1: 加载 ──────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("[清洗流水线] 开始 ...")
        print(f"[清洗流水线] 输入: data/raw/{input_filename}")
        print("=" * 60)

    df = load_raw_data(input_filename)
    report["rows_before"] = df.shape[0]
    report["cols_before"] = df.shape[1]

    # ── Step 2: 缺失值 ────────────────────────────────────────────
    if verbose:
        print("\n[1/5] 缺失值处理 ...")
    df = clean_missing_values(df)
    report["steps"].append("missing_values")

    # ── Step 3: 重复值 ────────────────────────────────────────────
    if verbose:
        print("\n[2/5] 重复值处理 ...")
    df = clean_duplicates(df)
    report["steps"].append("duplicates")

    # ── Step 4: 异常值 ────────────────────────────────────────────
    if verbose:
        print(f"\n[3/5] 异常值处理 (IQR × {iqr_multiplier}, 策略={outlier_strategy}) ...")
    df = clean_outliers(df, multiplier=iqr_multiplier, strategy=outlier_strategy)
    report["steps"].append("outliers")

    # ── Step 5: TotalCharges 一致性（在异常值盖帽之后）──────────
    if verbose:
        print("\n[4/5] TotalCharges 一致性检查 ...")
    df = clean_total_charges(df)
    report["steps"].append("total_charges_fix")

    # ── Step 6: 数据类型 ──────────────────────────────────────────
    if verbose:
        print("\n[5/5] 数据类型统一 ...")
    df = clean_data_types(df)
    report["steps"].append("data_types")

    # ── 汇总 ──────────────────────────────────────────────────────
    report["rows_after"] = df.shape[0]
    report["cols_after"] = df.shape[1]

    rows_removed = report["rows_before"] - report["rows_after"]
    if verbose:
        print("\n" + "=" * 60)
        print("[清洗流水线] 完成!")
        print(f"  行数: {report['rows_before']:,} → {report['rows_after']:,} "
              f"({rows_removed} 行被移除)")
        print(f"  列数: {report['cols_before']} → {report['cols_after']}")
        print("=" * 60)

    # ── 保存 ──────────────────────────────────────────────────────
    if save:
        output_path = save_processed(df, filename=output_filename)
        report["output_path"] = str(output_path)
        if verbose:
            print(f"\n[清洗流水线] 已保存至: {output_path}")

    return {
        "df_clean": df,
        "report": report,
        "output_path": report.get("output_path"),
    }


# ── 数据写出 ──────────────────────────────────────────────────────


def save_processed(
    df: pd.DataFrame,
    filename: str = "saas_customer_churn_cleaned.csv",
    directory: Optional[Path] = None,
) -> Path:
    """
    保存清洗后的数据到 data/processed/。

    Parameters
    ----------
    df : pd.DataFrame
        清洗后的数据。
    filename : str
        输出文件名。
    directory : Path, optional
        目标目录，默认 PROCESSED_DIR。

    Returns
    -------
    Path
    """
    target_dir = directory or PROCESSED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / filename
    df.to_csv(output_path, index=False)
    print(f"[clean] 已保存 {df.shape[0]:,} 行 × {df.shape[1]} 列 → {output_path}")
    return output_path


# ── 内存节省：将分类列转为 category 类型的辅助工具 ────────────────


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    内存优化：对高基数字符串列降低存储开销。

    会检测所有 object 列：
        - 若唯一值占比 < 50% → 转为 category
        - 数值列在安全范围内降级（float64 → float32）
    """
    df = df.copy()
    before_mb = df.memory_usage(deep=True).sum() / 1024**2

    # 字符串 → category
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category")

    # 浮点降级
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # 整数降级
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    after_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"[clean] 内存优化: {before_mb:.2f} MB → {after_mb:.2f} MB "
          f"(节省 {(1 - after_mb / before_mb) * 100:.1f}%)")
    return df


# ── CLI 测试入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print(">>> SaaS 客户流失数据清洗模块 — 自检")
    print("=" * 65)

    # ── 1. 完整流水线 ─────────────────────────────────────────────
    result = run_cleaning_pipeline(
        iqr_multiplier=1.5,
        outlier_strategy="cap",
        save=True,
    )

    df_clean = result["df_clean"]
    report = result["report"]

    # ── 2. 快速校验 ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(">>> 清洗结果校验")
    print("=" * 65)

    # 缺失值
    total_missing = df_clean.isnull().sum().sum()
    print(f"  缺失值: {total_missing} (应为 0)")

    # 重复值（排除 ID 列）
    dup_cols = [c for c in df_clean.columns if c != ID_COL]
    n_dup = df_clean[[c for c in dup_cols if c in df_clean.columns]].duplicated().sum()
    print(f"  重复行: {n_dup} (应为 0)")

    # TotalCharges 一致性
    if all(c in df_clean.columns for c in ["Tenure", "MonthlyCharges", "TotalCharges"]):
        computed = df_clean["MonthlyCharges"] * df_clean["Tenure"]
        diff = (df_clean["TotalCharges"] - computed).abs()
        max_diff = diff.max()
        print(f"  TotalCharges 最大偏差: {max_diff:.4f} (应接近 0)")

    # 异常值：检查 IQR 边界（TotalCharges 是衍生指标，不参与 IQR 盖帽）
    print("  异常值盖帽校验:")
    for col in ["MonthlyCharges", "SupportTickets",
                "LastLoginDays", "FeatureUsageCount"]:
        lower, upper, mask = detect_outliers_iqr(df_clean, col, multiplier=1.5)
        print(f"    {col:<22s} 范围=[{lower:.2f}, {upper:.2f}]  "
              f"越界={mask.sum()} (应为 0)")

    # 数据类型
    cat_count = len(df_clean.select_dtypes(include=["category"]).columns)
    print(f"  分类列: {cat_count} 个 category 类型")

    # 流失率
    if TARGET_COL in df_clean.columns:
        churn_rate = (df_clean[TARGET_COL] == "Yes").mean()
        print(f"  流失率: {churn_rate:.2%}")

    # ── 3. 内存优化 ───────────────────────────────────────────────
    print("\n> 内存优化 ...")
    df_opt = optimize_memory(df_clean)
    save_processed(df_opt, filename="saas_customer_churn_cleaned_opt.csv")

    print("\n" + "=" * 65)
    print(">>> 自检完成，清洗模块运行正常。")
    print("=" * 65)
