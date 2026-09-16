# -*- coding: utf-8 -*-
"""
資料集讀取範例
建議使用 Polars 讀取 Parquet（速度快、記憶體省）
"""

# =============================================
#  方法一：Polars（推薦）
# =============================================
# pip install polars
from pathlib import Path

import polars as pl

PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_DIR / "來源資料" / "visibility_satellite_2023_clean.parquet"

# 讀取全部資料（~1秒，記憶體 ~300MB）
df = pl.read_parquet(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(df.head())

# 讀取部分欄位（更快更省記憶體）
df_small = pl.read_parquet(
    DATA_PATH,
    columns=["Station_ID", "DateTime_UTC0", "Visibility_km", "B13", "B14", "B15"]
)

# Lazy 模式：適合大資料篩選，不會一次全讀進記憶體
df_fog = (
    pl.scan_parquet(DATA_PATH)
    .filter(pl.col("is_fog") == 1)
    .collect()
)
print(f"Fog events: {df_fog.shape[0]}")


# =============================================
#  方法二：Pandas（可用，但較慢較耗記憶體）
# =============================================
# pip install pandas pyarrow
import pandas as pd

# 讀 Parquet（~3-5秒，記憶體 ~800MB-1GB）
df_pd = pd.read_parquet(DATA_PATH)
print(f"Shape: {df_pd.shape}")
print(f"Memory: {df_pd.memory_usage(deep=True).sum() / 1024**2:.0f} MB")


# =============================================
#  常用操作範例
# =============================================

# --- 取出特定測站 ---
station_c01 = df.filter(pl.col("Station_ID") == "C01")

# --- 取出特定月份 ---
march = df.filter(pl.col("DateTime_UTC0").dt.month() == 3)

# --- 分站統計霧事件 ---
fog_stats = (
    df.group_by("Station_ID")
    .agg([
        pl.len().alias("total"),
        pl.col("is_fog").sum().alias("fog_count"),
        (pl.col("is_fog").sum() / pl.len() * 100).alias("fog_pct"),
    ])
    .sort("fog_pct", descending=True)
)
print(fog_stats)

# --- 準備模型輸入 (X, y) ---
feature_cols = [f"B{i:02d}" for i in range(1, 17)]  # B01~B16
X = df.select(feature_cols).to_numpy()
y = df["is_fog"].to_numpy()
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Fog ratio: {y.mean():.4f}")

# =============================================
#  建議衍生特徵（完整版，詳見 README.md）
# =============================================
import numpy as np

df_features = df.with_columns([
    # --- A. 衛星衍生特徵（經 KS 檢定驗證有效）---
    # Split Window Difference：霧偵測最關鍵的指標
    (pl.col("B14") - pl.col("B15")).alias("BTD_B14_B15"),
    # 紅外窗區差異
    (pl.col("B13") - pl.col("B15")).alias("BTD_B13_B15"),
    # 近紅外/可見光比值（僅白天有意義）
    (pl.col("B04") / (pl.col("B02") + 1.0)).alias("B04_B02_ratio"),
    # 可見光比值
    (pl.col("B03") / (pl.col("B04") + 0.001)).alias("B03_B04_ratio"),
    # 水氣通道差（反映大氣水氣垂直分布）
    (pl.col("B08") - pl.col("B10")).alias("WV_diff_B08_B10"),
    (pl.col("B09") - pl.col("B10")).alias("WV_diff_B09_B10"),

    # --- B. 時間特徵 ---
    # 日週期（sin/cos 編碼避免 23→0 的不連續）
    (pl.col("DateTime_UTC0").dt.hour() * 2.0 * np.pi / 24.0).sin().alias("hour_sin"),
    (pl.col("DateTime_UTC0").dt.hour() * 2.0 * np.pi / 24.0).cos().alias("hour_cos"),
    # 季節週期
    (pl.col("DateTime_UTC0").dt.month() * 2.0 * np.pi / 12.0).sin().alias("month_sin"),
    (pl.col("DateTime_UTC0").dt.month() * 2.0 * np.pi / 12.0).cos().alias("month_cos"),
])

# --- 太陽高度角（判斷日夜，比固定時間切分更準確）---
# 完整公式見 AI/add_features.py 的 solar_elevation()
# 簡化版：用 UTC 時間 + 經度推估
hour_utc = df["DateTime_UTC0"].dt.hour() + df["DateTime_UTC0"].dt.minute() / 60.0
local_solar_hour = hour_utc + df["Longitude"] / 15.0  # 粗略太陽時
# 太陽高度角 < 0 → 夜間
# 精確計算需考慮赤緯與日期，詳見 add_features.py


# =============================================
#  LOMOCV 交叉驗證範例
# =============================================
# 推薦的驗證策略：Leave-One-Month-Out
# 每次留 1 個月當測試集，其餘 11 個月訓練

from sklearn.metrics import roc_auc_score

feature_cols = [f"B{i:02d}" for i in range(1, 17)]
# 可以加入更多特徵：
# feature_cols += ["Elevation_m", "Latitude", "Longitude",
#                  "BTD_B14_B15", "WV_diff_B08_B10", "hour_sin", "hour_cos"]

df_with_month = df_features.with_columns(
    pl.col("DateTime_UTC0").dt.month().alias("month")
)

for test_month in range(1, 13):
    train = df_with_month.filter(pl.col("month") != test_month)
    test = df_with_month.filter(pl.col("month") == test_month)

    X_train = train.select(feature_cols).to_numpy()
    y_train = train["is_fog"].to_numpy()
    X_test = test.select(feature_cols).to_numpy()
    y_test = test["is_fog"].to_numpy()

    # 在這裡訓練你的模型
    # model.fit(X_train, y_train)
    # y_prob = model.predict_proba(X_test)[:, 1]
    # auc = roc_auc_score(y_test, y_prob)
    print(f"Month {test_month:2d}: train={len(y_train):>8,}, test={len(y_test):>7,}, "
          f"fog_in_test={y_test.sum():>5.0f} ({y_test.mean()*100:.2f}%)")
