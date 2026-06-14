"""
Extract — 数据提取层。

从 CSV 文件加载清洗后的客户数据。
"""

from __future__ import annotations

import pandas as pd

from .config import DATA_DIR, DEFAULT_DATAFILE


def extract(filename: str | None = None) -> pd.DataFrame:
    """
    从 data/processed/ 提取清洗后的数据。

    Parameters
    ----------
    filename : str, optional
        数据文件名，默认使用 DEFAULT_DATAFILE。

    Returns
    -------
    pd.DataFrame
        原始客户数据（包含 CustomerID）。
    """
    path = DATA_DIR / (filename or DEFAULT_DATAFILE)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path)
    print(f"[Extract] Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols  "
          f"from {path}")
    return df


if __name__ == "__main__":
    df = extract()
    print(df.head(3))
