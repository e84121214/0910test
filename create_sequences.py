import polars as pl
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = "visibility_satellite_2023_clean.parquet"

STATION = "C19"

LOOKBACK = 18
HORIZON = 6

EXPERIMENT = "SAT"

SATELLITE_FEATURES = [
    "B01", "B02", "B03", "B04",
    "B05", "B06", "B07", "B08",
    "B09", "B10", "B11", "B12",
    "B13", "B14", "B15", "B16",
]

VISIBILITY_FEATURES = ["Visibility_km"]

if EXPERIMENT == "SAT":
    FEATURES = SATELLITE_FEATURES
elif EXPERIMENT == "VIS":
    FEATURES = VISIBILITY_FEATURES
elif EXPERIMENT == "SAT_VIS":
    FEATURES = SATELLITE_FEATURES + VISIBILITY_FEATURES
else:
    raise ValueError(f"未知的 EXPERIMENT：{EXPERIMENT}")

TARGET = "Visibility_km"

print("實驗模式：", EXPERIMENT)
print("輸入 Features：", FEATURES)
print("Feature 數量：", len(FEATURES))


# =========================================================
# 1. 讀取資料
# =========================================================

df = pl.read_parquet(DATA_PATH)

station_df = (
    df.filter(pl.col("Station_ID") == STATION)
      .sort("DateTime_UTC0")
)

print("原始測站資料筆數：", station_df.height)


# =========================================================
# 2. 建立連續區段 segment_id
# =========================================================

station_df = station_df.with_columns(
    pl.col("DateTime_UTC0")
      .diff()
      .alias("time_diff")
)

station_df = station_df.with_columns(
    (
        pl.col("time_diff") != pl.duration(minutes=10)
    )
    .fill_null(True)
    .cum_sum()
    .alias("segment_id")
)


# =========================================================
# 3. 建立 sequence
# =========================================================

def create_sequences(df, features, target, lookback, horizon):
    X_list = []
    y_list = []
    target_time_list = []

    persistence_list = []

    segments = df.partition_by("segment_id")

    for segment in segments:
        n = segment.height

        # 這個 segment 太短，無法產生樣本
        if n < lookback + horizon:
            continue

        feature_array = segment.select(features).to_numpy()
        target_array = segment[target].to_numpy()
        time_array = segment["DateTime_UTC0"].to_numpy()

        for i in range(n - lookback - horizon + 1):
            X = feature_array[i:i + lookback]

            target_index = i + lookback + horizon - 1
            y = target_array[target_index]

            target_time = time_array[target_index]
            current_index = i + lookback - 1
            persistence_value = target_array[current_index]
            
            X_list.append(X)
            y_list.append(y)
            target_time_list.append(target_time)

            persistence_list.append(persistence_value)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    target_times = np.array(target_time_list)

    persistence = np.array(
        persistence_list,
        dtype=np.float32
    )

    return X, y, target_times, persistence


X, y, target_times, persistence = create_sequences(
    station_df,
    FEATURES,
    TARGET,
    LOOKBACK,
    HORIZON
)


# =========================================================
# 4. 檢查結果
# =========================================================

print("\n=== Sequence 建立完成 ===")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("target_times shape:", target_times.shape)

print("\n第一筆 X shape:", X[0].shape)
print("第一筆 y:", y[0])
print("第一筆 target time:", target_times[0])

# =========================================================
# 5. 依時間切分 Train / Validation / Test
# =========================================================

train_end = np.datetime64("2023-10-01")
val_end = np.datetime64("2023-11-01")

train_mask = target_times < train_end

val_mask = (
    (target_times >= train_end)
    & (target_times < val_end)
)

test_mask = target_times >= val_end


X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

persistence_train = persistence[train_mask]
persistence_val = persistence[val_mask]
persistence_test = persistence[test_mask]

# =========================================================
# 5.1 移除含 NaN 的 sequence
# =========================================================

train_valid_mask = (
    ~np.isnan(X_train).any(axis=(1, 2))
    & ~np.isnan(y_train)
    & ~np.isnan(persistence_train)
)
val_valid_mask = (
    ~np.isnan(X_val).any(axis=(1, 2))
    & ~np.isnan(y_val)
    & ~np.isnan(persistence_val)
)
test_valid_mask = (
    ~np.isnan(X_test).any(axis=(1, 2))
    & ~np.isnan(y_test)
    & ~np.isnan(persistence_test)
)

print("\n=== 移除 NaN 前 ===")
print("Train 含 NaN sequence 數：", (~train_valid_mask).sum())
print("Validation 含 NaN sequence 數：", (~val_valid_mask).sum())
print("Test 含 NaN sequence 數：", (~test_valid_mask).sum())

X_train = X_train[train_valid_mask]
y_train = y_train[train_valid_mask]

persistence_train = persistence_train[
    train_valid_mask
]

persistence_val = persistence_val[
    val_valid_mask
]

persistence_test = persistence_test[
    test_valid_mask
]

X_val = X_val[val_valid_mask]
y_val = y_val[val_valid_mask]

X_test = X_test[test_valid_mask]
y_test = y_test[test_valid_mask]

print("\n=== 移除 NaN 後 ===")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

print("Train NaN 數量：", np.isnan(X_train).sum())
print("Validation NaN 數量：", np.isnan(X_val).sum())
print("Test NaN 數量：", np.isnan(X_test).sum())

print("\n=== Train / Validation / Test ===")

print("Train:")
print("X:", X_train.shape)
print("y:", y_train.shape)

print("\nValidation:")
print("X:", X_val.shape)
print("y:", y_val.shape)

print("\nTest:")
print("X:", X_test.shape)
print("y:", y_test.shape)

# NaN 檢查
print("\n=== NaN 檢查 ===")
print("X NaN 數量：", np.isnan(X).sum())
print("y NaN 數量：", np.isnan(y).sum())

# 檢查每個 feature 的 NaN 數量
nan_per_feature = np.isnan(X).sum(axis=(0, 1))

print("\n=== 每個 Feature 的 NaN 數量 ===")

for feature, nan_count in zip(FEATURES, nan_per_feature):
    print(f"{feature}: {nan_count}")

# =========================================================
# 6. 使用 Training Set 統計量進行 Standardization
# =========================================================

feature_mean = X_train.mean(axis=(0, 1), keepdims=True)
feature_std = X_train.std(axis=(0, 1), keepdims=True)

# 避免某個 feature 標準差剛好為 0
feature_std[feature_std == 0] = 1.0


X_train_scaled = (
    X_train - feature_mean
) / feature_std

X_val_scaled = (
    X_val - feature_mean
) / feature_std

X_test_scaled = (
    X_test - feature_mean
) / feature_std


print("\n=== Standardization 完成 ===")

print(
    "Train standardized mean:",
    X_train_scaled.mean(axis=(0, 1))
)

print(
    "Train standardized std:",
    X_train_scaled.std(axis=(0, 1))
)

# =========================================================
# 7. NumPy → PyTorch Tensor
# =========================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).unsqueeze(1)


X_val_tensor = torch.tensor(
    X_val_scaled,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.float32
).unsqueeze(1)


X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).unsqueeze(1)


print("\n=== Tensor shape ===")

print(
    "X_train:",
    X_train_tensor.shape
)

print(
    "y_train:",
    y_train_tensor.shape
)

# =========================================================
# 8. 建立 DataLoader
# =========================================================

BATCH_SIZE = 128

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


first_X, first_y = next(iter(train_loader))

print("\n=== 第一個 Batch ===")
print("X batch shape:", first_X.shape)
print("y batch shape:", first_y.shape)
