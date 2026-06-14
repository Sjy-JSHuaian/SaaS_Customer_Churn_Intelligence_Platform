"""
数据子包 — SaaS 客户流失预测系统的数据 I/O 层。

提供加载、清洗、写出全链路功能。

加载:
    - load_raw()              一键加载 data/raw/ 中的默认数据集
    - load_csv()              从指定 CSV 路径加载
    - load_from_url()         从远程 URL 下载并加载
    - generate_sample_data()  生成 Demo / CI 用样本数据

清洗:
    - load_raw_data()         加载原始数据（预处理专用入口）
    - clean_missing_values()  缺失值补全
    - clean_duplicates()      重复行删除
    - clean_outliers()        异常值盖帽
    - clean_total_charges()   TotalCharges 一致性修复
    - clean_data_types()      数据类型统一
    - optimize_memory()       内存优化
    - run_cleaning_pipeline() 一键执行完整清洗流水线

写出 / 探查:
    - save_to_csv()           保存到 data/raw/
    - save_to_processed()     保存到 data/processed/
    - get_data_info()         快速探查数据结构
    - preview()               查看前 N 行
    - get_column_groups()     按类型分组列名
"""

from .loader import (
    # 常量
    PROJECT_ROOT,
    RAW_DIR,
    PROCESSED_DIR,
    COLUMN_MAP,
    DEFAULT_FILENAME,
    # 加载
    load_raw,
    load_csv,
    load_from_url,
    generate_sample_data,
    # 写出
    save_to_csv,
    save_to_processed,
    # 探查
    get_data_info,
    preview,
    get_column_groups,
)

from .preprocess import (
    load_raw_data,
    clean_missing_values,
    clean_duplicates,
    clean_outliers,
    clean_total_charges,
    clean_data_types,
    detect_outliers_iqr,
    optimize_memory,
    run_cleaning_pipeline,
    save_processed,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
)

__all__ = [
    # 路径常量
    "PROJECT_ROOT",
    "RAW_DIR",
    "PROCESSED_DIR",
    "COLUMN_MAP",
    "DEFAULT_FILENAME",
    # 加载
    "load_raw",
    "load_csv",
    "load_from_url",
    "generate_sample_data",
    # 清洗
    "load_raw_data",
    "clean_missing_values",
    "clean_duplicates",
    "clean_outliers",
    "clean_total_charges",
    "clean_data_types",
    "detect_outliers_iqr",
    "optimize_memory",
    "run_cleaning_pipeline",
    # 写出
    "save_to_csv",
    "save_to_processed",
    "save_processed",
    # 探查
    "get_data_info",
    "preview",
    "get_column_groups",
    # 列名常量
    "NUMERICAL_COLS",
    "CATEGORICAL_COLS",
    "TARGET_COL",
]
