"""
数据加载模块 — SaaS 客户流失预测系统。

职责：
    1. 从 CSV / URL / 编程生成方式加载原始客户流失数据。
    2. 提供统一的数据写入接口（存储到 data/raw/ 或 data/processed/）。
    3. 提供快速探查工具（info、preview、missing 统计）。

设计原则：
    - 单一职责：loader 只负责 I/O，不做特征工程。
    - 可复用：更换数据源时只需新增加载函数，其他模块不受影响。
    - 符合工程规范：模块化、有类型标注、有 CLI 自检入口。

支持的数据源：
    - CSV 文件（放于 data/raw/ 目录下）
    - URL 远程下载
    - 内置生成函数（用于 Demo / CI / 快速原型）
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# ── 项目路径常量 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 确保目录存在
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── SaaS 客户流失数据集 — 标准列名映射 ──────────────────────────
# 中文列名 → 英文标准列名（兼容不同来源的数据文件）
COLUMN_MAP: dict[str, str] = {
    # 客户标识
    "客户ID":       "CustomerID",
    "customerid":   "CustomerID",
    "customer_id":  "CustomerID",

    # 人口统计
    "性别":         "Gender",
    "是否为老年人": "SeniorCitizen",
    "是否有伴侣":   "Partner",
    "是否有家属":   "Dependents",

    # 服务订阅
    "使用时长":     "Tenure",
    "月费":         "MonthlyCharges",
    "总费用":       "TotalCharges",
    "合同类型":     "Contract",
    "支付方式":     "PaymentMethod",
    "电子账单":     "PaperlessBilling",

    # 附加服务
    "电话服务":     "PhoneService",
    "多线路":       "MultipleLines",
    "互联网服务":   "InternetService",
    "在线安全":     "OnlineSecurity",
    "在线备份":     "OnlineBackup",
    "设备保护":     "DeviceProtection",
    "技术支持":     "TechSupport",
    "流媒体电视":   "StreamingTV",
    "流媒体电影":   "StreamingMovies",

    # SaaS 特有字段
    "工单数":       "SupportTickets",
    "最近登录":     "LastLoginDays",
    "功能使用数":   "FeatureUsageCount",
    "公司规模":     "CompanySize",
    "行业":         "Industry",

    # 目标
    "流失":         "Churn",
    "是否流失":     "Churn",
    "churn":        "Churn",
}

# 数据集默认文件名
DEFAULT_FILENAME = "saas_customer_churn_raw.csv"

# ── 核心加载函数 ──────────────────────────────────────────────────


def load_csv(
    filepath: Union[str, Path],
    *,
    normalize_columns: bool = True,
    target_col: Optional[str] = "Churn",
    drop_id: bool = True,
) -> pd.DataFrame:
    """
    从本地 CSV 文件加载 SaaS 客户流失数据。

    自动完成以下处理：
        1. 列名标准化（中文 → 英文，统一大小写）
        2. 删除高基数 ID 列（可关闭）
        3. 目标列识别日志

    Parameters
    ----------
    filepath : str or Path
        CSV 文件路径。
    normalize_columns : bool, default=True
        是否自动标准化列名（中文映射、去空格、统一大小写）。
    target_col : str, optional
        目标列名称，仅用于校验日志。传 None 跳过校验。
    drop_id : bool, default=True
        是否删除 CustomerID 列。

    Returns
    -------
    pd.DataFrame
        清洗后的客户流失数据。

    Raises
    ------
    FileNotFoundError
        文件不存在时抛出。
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path.resolve()}")

    df = pd.read_csv(path)

    # 列名标准化
    if normalize_columns:
        df = _normalize_columns(df)

    # 删除 ID 列
    if drop_id:
        id_cols = [c for c in df.columns if c.lower() in ("customerid", "customer_id")]
        if id_cols:
            df = df.drop(columns=id_cols)
            print(f"[loader] 已删除 ID 列: {id_cols}")

    # 目标列校验
    if target_col and target_col in df.columns:
        churn_counts = df[target_col].value_counts().to_dict()
        print(f"[loader] 目标列 '{target_col}' 分布: {churn_counts}")
    elif target_col and target_col not in df.columns:
        print(f"[loader] ⚠ 目标列 '{target_col}' 不在数据集中")

    print(f"[loader] 加载完成: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    return df


def load_from_url(
    url: str,
    *,
    normalize_columns: bool = True,
    drop_id: bool = True,
    save_raw: bool = True,
    raw_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    从远程 URL 下载 SaaS 客户流失数据。

    下载后自动保存到 data/raw/ 目录，下次可直接用 load_csv() 读取。

    Parameters
    ----------
    url : str
        数据文件的远程 URL（CSV 格式）。
    normalize_columns : bool, default=True
        是否自动标准化列名。
    drop_id : bool, default=True
        是否删除 ID 列。
    save_raw : bool, default=True
        是否将下载的文件保存到 data/raw/。
    raw_filename : str, optional
        保存的文件名，默认使用内置文件名。

    Returns
    -------
    pd.DataFrame
    """
    print(f"[loader] 正在从 URL 下载数据: {url}")
    df = pd.read_csv(url)
    print(f"[loader] 下载完成: {df.shape[0]:,} 行 × {df.shape[1]} 列")

    if normalize_columns:
        df = _normalize_columns(df)

    if drop_id:
        id_cols = [c for c in df.columns if c.lower() in ("customerid", "customer_id")]
        if id_cols:
            df = df.drop(columns=id_cols)

    if save_raw:
        filename = raw_filename or DEFAULT_FILENAME
        save_to_csv(df, filename=filename)

    return df


def load_raw(
    filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    从 data/raw/ 目录加载默认数据集。

    这是项目中最常用的入口——只需调用 load_raw() 即可获取标准数据。

    Parameters
    ----------
    filename : str, optional
        文件名，默认使用 DEFAULT_FILENAME。

    Returns
    -------
    pd.DataFrame
    """
    filename = filename or DEFAULT_FILENAME
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"默认数据集不存在: {path}\n"
            f"请将 CSV 文件放置到 data/raw/ 目录，"
            f"或使用 load_csv() 指定路径。"
        )
    return load_csv(path)


# ── 内置数据生成（Demo / CI / 快速原型）──────────────────────────


def generate_sample_data(
    n_samples: int = 7043,
    random_state: int = 42,
    save: bool = True,
) -> pd.DataFrame:
    """
    生成一份符合真实分布的 SaaS 客户流失样本数据。

    数据包含以下特征组：
        - 人口统计：Gender, SeniorCitizen, Partner, Dependents
        - 订阅服务：Tenure, MonthlyCharges, Contract, PaymentMethod, PaperlessBilling
        - 附加服务：PhoneService, InternetService, OnlineSecurity 等
        - SaaS 特有：SupportTickets, LastLoginDays, FeatureUsageCount
        - 目标变量：Churn（≈26.5% 正类，接近行业真实比例）

    Parameters
    ----------
    n_samples : int, default=7043
        样本数量（默认 7043，与经典 Telco 数据集一致）。
    random_state : int, default=42
        随机种子，保证可复现。
    save : bool, default=True
        是否保存到 data/raw/。

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(random_state)

    # ── 人口统计 ──────────────────────────────────────────────────
    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.50, 0.50])
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])

    # ── 订阅 & 计费 ──────────────────────────────────────────────
    tenure = rng.integers(0, 73, size=n_samples)  # 0-72 个月
    monthly_charges = np.round(rng.uniform(18.25, 118.75, size=n_samples), 2)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples, p=[0.55, 0.24, 0.21],
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples, p=[0.34, 0.23, 0.22, 0.21],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])

    # ── 服务 ─────────────────────────────────────────────────────
    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "Yes",
        rng.choice(["Yes", "No", "No phone service"], size=n_samples, p=[0.42, 0.48, 0.10]),
        "No phone service",
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_samples, p=[0.35, 0.44, 0.21],
    )
    online_security = rng.choice(["Yes", "No", "No internet service"], size=n_samples, p=[0.29, 0.50, 0.21])
    online_backup = rng.choice(["Yes", "No", "No internet service"], size=n_samples, p=[0.30, 0.49, 0.21])
    device_protection = rng.choice(["Yes", "No", "No internet service"], size=n_samples, p=[0.30, 0.49, 0.21])
    tech_support = rng.choice(["Yes", "No", "No internet service"], size=n_samples, p=[0.30, 0.49, 0.21])
    streaming_tv = rng.choice(["Yes", "No", "No internet service"], size=n_samples, p=[0.32, 0.47, 0.21])
    streaming_movies = rng.choice(["Yes", "No", "No internet service"], size=n_samples, p=[0.32, 0.47, 0.21])

    # ── SaaS 特有字段 ────────────────────────────────────────────
    support_tickets = rng.poisson(lam=1.2, size=n_samples).clip(0, 9)
    last_login_days = rng.exponential(scale=8.0, size=n_samples).clip(0, 60).astype(int)
    feature_usage_count = rng.binomial(n=12, p=0.55, size=n_samples)
    company_size = rng.choice(
        ["Micro (<10)", "Small (10-50)", "Medium (51-250)", "Large (251-1000)", "Enterprise (1000+)"],
        size=n_samples, p=[0.20, 0.30, 0.28, 0.15, 0.07],
    )
    industry = rng.choice(
        ["Technology", "Finance", "Healthcare", "Education", "Retail",
         "Manufacturing", "Media", "Real Estate", "Consulting", "Other"],
        size=n_samples,
    )

    # ── 目标变量（基于真实模式构造）──────────────────────────────
    # 短期 + 月付 → 高流失率；长期 + 年付 → 低流失率
    churn_prob = np.where(
        contract == "Month-to-month",
        0.45 - tenure * 0.003,     # 月付：初期 ~45%，逐年递减
        np.where(
            contract == "One year",
            0.15 - tenure * 0.002,  # 年付：初期 ~15%
            0.05 - tenure * 0.001,  # 两年付：初期 ~5%
        ),
    )
    # 支持工单多 → 流失上升
    churn_prob += support_tickets * 0.035
    # 长期不登录 → 流失上升
    churn_prob += (last_login_days > 20).astype(float) * 0.15
    # 低功能使用 → 流失上升
    churn_prob += (feature_usage_count < 3).astype(float) * 0.10

    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = rng.binomial(1, churn_prob)
    churn_label = np.where(churn == 1, "Yes", "No")

    # ── 组装 DataFrame ───────────────────────────────────────────
    df = pd.DataFrame({
        "CustomerID":       [f"{i:04d}-XXXXX" for i in range(n_samples)],
        "Gender":           gender,
        "SeniorCitizen":    senior_citizen,
        "Partner":          partner,
        "Dependents":       dependents,
        "Tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     np.round(monthly_charges * tenure, 2),
        "Contract":         contract,
        "PaymentMethod":    payment_method,
        "PaperlessBilling": paperless_billing,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "SupportTickets":   support_tickets,
        "LastLoginDays":    last_login_days,
        "FeatureUsageCount": feature_usage_count,
        "CompanySize":      company_size,
        "Industry":         industry,
        "Churn":            churn_label,
    })

    churn_rate = (churn_label == "Yes").mean()
    print(f"[loader] 生成样本数据: {n_samples:,} 行 × {df.shape[1]} 列")
    print(f"[loader] 流失率: {churn_rate:.2%}")

    if save:
        save_to_csv(df, filename=DEFAULT_FILENAME)
        print(f"[loader] 已保存至: {RAW_DIR / DEFAULT_FILENAME}")

    return df


# ── 数据写出 ──────────────────────────────────────────────────────


def save_to_csv(
    df: pd.DataFrame,
    filename: str = DEFAULT_FILENAME,
    directory: Optional[Path] = None,
) -> Path:
    """
    将 DataFrame 保存到指定目录。

    Parameters
    ----------
    df : pd.DataFrame
        要保存的数据。
    filename : str
        输出文件名。
    directory : Path, optional
        目标目录，默认为 RAW_DIR。

    Returns
    -------
    Path
        实际写入的文件路径。
    """
    target_dir = directory or RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / filename
    df.to_csv(output_path, index=False)
    print(f"[loader] 已保存 {df.shape[0]:,} 行 → {output_path}")
    return output_path


def save_to_processed(
    df: pd.DataFrame,
    filename: str,
) -> Path:
    """
    保存加工后的数据到 data/processed/ 目录。

    Parameters
    ----------
    df : pd.DataFrame
        加工后的数据。
    filename : str
        输出文件名。

    Returns
    -------
    Path
    """
    return save_to_csv(df, filename=filename, directory=PROCESSED_DIR)


# ── 数据探查工具 ──────────────────────────────────────────────────


def get_data_info(df: pd.DataFrame) -> dict:
    """
    获取 DataFrame 的基本统计信息，用于快速数据探查。

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        包含 shape, dtypes, missing, describe, head 等信息。
    """
    # 分离数值列和非数值列的统计
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "num_columns": num_cols,
        "cat_columns": cat_cols,
        "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "describe_num": df.describe().round(4) if num_cols else None,
        "describe_cat": df.describe(include=["object", "string"]) if cat_cols else None,
        "head": df.head(3),
    }


def preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """返回前 n 行，方便在 Notebook / CLI 中快速查看。"""
    return df.head(n)


def get_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    按类型和业务含义对列进行分组，帮助后续特征工程。

    Returns
    -------
    dict
        包含 categorical, numerical, binary, target 分组。
    """
    cats = df.select_dtypes(exclude=[np.number]).columns.tolist()
    nums = df.select_dtypes(include=[np.number]).columns.tolist()

    target_candidates = [c for c in cats if c.lower() in ("churn", "churn_label")]
    binary_candidates = [c for c in cats if df[c].nunique() == 2 and c not in target_candidates]

    return {
        "categorical": [c for c in cats if c not in target_candidates],
        "numerical": nums,
        "binary": binary_candidates,
        "target": target_candidates,
    }


# ── 内部辅助函数 ──────────────────────────────────────────────────


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    列名标准化流水线：
        1. 去掉首尾空格
        2. 应用 COLUMN_MAP 映射（中文 → 英文）
        3. 统一为首字母大写
    """
    df = df.rename(columns=lambda c: str(c).strip())
    df = df.rename(columns=COLUMN_MAP)
    df = df.rename(columns=lambda c: (
        c.strip().replace(" ", "").replace("_", " ").title().replace(" ", "")
        if any("一" <= ch <= "鿿" for ch in c) else c
    ))
    return df


# ── CLI 测试入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print(">>> SaaS 客户流失数据加载模块 — 自检")
    print("=" * 65)

    # 1. 生成样本数据
    print("\n[1] 生成样本数据 ...")
    df = generate_sample_data(n_samples=500, random_state=42, save=False)
    print(preview(df, n=3))

    # 2. 数据信息
    print("\n[2] 数据探查 ...")
    info = get_data_info(df)
    print(f"    形状: {info['shape']}")
    print(f"    数值列 ({len(info['num_columns'])}): {info['num_columns']}")
    print(f"    分类列 ({len(info['cat_columns'])}): {info['cat_columns']}")
    print(f"    缺失值比例 (%): {info['missing_pct']}")

    # 3. 列分组
    print("\n[3] 列分组 ...")
    groups = get_column_groups(df)
    for k, v in groups.items():
        print(f"    {k}: {v}")

    # 4. 保存到 data/raw/
    print("\n[4] 保存原始数据 ...")
    saved = save_to_csv(df, filename=DEFAULT_FILENAME)
    print(f"    已保存至: {saved}")

    # 5. 验证读取
    print("\n[5] 验证 load_csv 读取 ...")
    df_loaded = load_csv(saved)
    assert df_loaded.shape == df.drop(columns=["CustomerID"]).shape, \
        f"读取后形状不一致: {df_loaded.shape} vs {df.shape}"
    print("    [OK] 读写一致性校验通过")

    print("\n" + "=" * 65)
    print(">>> 自检完成，模块运行正常。")
    print("=" * 65)
