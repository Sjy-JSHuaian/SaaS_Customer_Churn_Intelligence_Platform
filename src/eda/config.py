"""
EDA 共享配置 — 路径、样式、色板。

所有子模块通过 `from .config import *` 获取统一的：
    - 路径常量 (RAW_DIR, FIGURES_DIR)
    - matplotlib / seaborn 全局样式
    - 品牌色板 (CHURN_COLORS, CHURN_PALETTE)
    - 保存辅助函数 (_save_fig)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ── 路径常量 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# 确保目录存在
RAW_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── matplotlib / seaborn 全局样式 ─────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
sns.set_style("whitegrid")
sns.set_palette("muted")

# ── 品牌色板（流失/留存专用）───────────────────────────────────────
CHURN_COLORS = {"Yes": "#E74C3C", "No": "#2ECC71"}
CHURN_PALETTE = [CHURN_COLORS["No"], CHURN_COLORS["Yes"]]

# ── 默认数据文件 ──────────────────────────────────────────────────
DEFAULT_DATAFILE = "saas_customer_churn_raw.csv"


# ── 保存辅助函数 ──────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str) -> Path:
    """保存图表到 reports/figures/，同时输出 PNG 和 SVG。"""
    for ext in [".png", ".svg"]:
        path = FIGURES_DIR / f"{name}{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"  [OK] Saved: {path}")
    plt.close(fig)
    return FIGURES_DIR / f"{name}.png"


def load_data() -> "pd.DataFrame":
    """加载原始数据集。"""
    import pandas as pd
    path = RAW_DIR / DEFAULT_DATAFILE
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    print(f"[EDA] Loaded data: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def print_section_header(title: str, step: str = "") -> None:
    """打印统一的节标题。"""
    print("\n" + "=" * 60)
    if step:
        print(f">>> [{step}] {title}")
    else:
        print(f">>> {title}")
    print("=" * 60)
