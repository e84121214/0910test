"""Build per-station fog sequences for one shared GRU.

Temporal test uses training stations in June/December. Spatial test uses the
held-out station in training months, so only the station changes.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parents[1]
DATA_PATH = PROJECT_DIR / "來源資料" / "visibility_satellite_2023_btd.parquet"
DATA_SOURCE = "ACOS"  # ACOS stations have the same stated visibility ceiling
SPATIAL_TEST_STATION = "C48"
LOOKBACK = 18
HORIZON = 6
FOG_THRESHOLD_KM = 1.0
SATELLITE_FEATURES = [f"B{i:02d}" for i in range(1, 17)]
BAND_DIFFERENCE_FEATURES = [
    "B07_minus_B14", "B07_minus_B13", "B11_minus_B15",
    "B13_minus_B15", "B14_minus_B15", "B15_minus_B16",
]
TIME_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
GEO_FEATURES = ["Latitude", "Longitude", "Elevation_m"]
FEATURE_SETS = {
    "SAT": SATELLITE_FEATURES + BAND_DIFFERENCE_FEATURES + TIME_FEATURES + GEO_FEATURES,
}
VALIDATION_MONTHS = {3, 9}
TEMPORAL_TEST_MONTHS = {6, 12}
BATCH_SIZE = 128


class SequenceDataset(Dataset):
    def __init__(self, segments, records, mean, std):
        self.segments = segments
        self.records = records  # segment index, window start, target index
        self.mean = mean
        self.std = std
        self.labels = np.asarray(
            [segments[s]["visibility"][t] < FOG_THRESHOLD_KM for s, _, t in records],
            dtype=np.float32,
        )
        self.persistence = np.asarray(
            [segments[s]["visibility"][start + LOOKBACK - 1] for s, start, _ in records],
            dtype=np.float32,
        )
        self.station_ids = np.asarray([segments[s]["station"] for s, _, _ in records])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        segment_id, start, _ = self.records[index]
        window = self.segments[segment_id]["features"][start:start + LOOKBACK]
        x = torch.from_numpy(((window - self.mean) / self.std).astype(np.float32))
        y = torch.tensor([self.labels[index]], dtype=torch.float32)
        return x, y


def build_datasets(feature_mode="SAT"):
    if feature_mode not in FEATURE_SETS:
        raise ValueError(f"未知的特徵組合：{feature_mode}")
    feature_names = FEATURE_SETS[feature_mode]
    columns = ["Station_ID", "Data_Source", "DateTime_UTC0", "Visibility_km",
               *[name for name in feature_names if name not in TIME_FEATURES]]
    missing = set(columns) - set(pl.read_parquet_schema(DATA_PATH))
    if missing:
        raise ValueError(f"Missing required columns in {DATA_PATH}: {sorted(missing)}")
    df = pl.read_parquet(DATA_PATH, columns=columns).filter(pl.col("Data_Source") == DATA_SOURCE)
    local_time = pl.col("DateTime_UTC0") + pl.duration(hours=8)
    hour_angle = (local_time.dt.hour() + local_time.dt.minute() / 60) * (2 * np.pi / 24)
    month_angle = (local_time.dt.month() - 1) * (2 * np.pi / 12)
    df = df.with_columns(
        hour_angle.sin().alias("hour_sin"), hour_angle.cos().alias("hour_cos"),
        month_angle.sin().alias("month_sin"), month_angle.cos().alias("month_cos"),
    )
    station_ids = sorted(df["Station_ID"].unique().to_list())
    if SPATIAL_TEST_STATION not in station_ids or len(station_ids) < 2:
        raise ValueError("空間測試站不存在，或可用測站不足兩站")
    print(f"特徵組合：{feature_mode} ({len(feature_names)} features)；資料來源：{DATA_SOURCE}；"
          f"測站數：{len(station_ids)}；保留站：{SPATIAL_TEST_STATION}")

    segments = []
    split_records = {name: [] for name in ("train", "val", "temporal_test", "spatial_test")}
    for station_id in station_ids:
        station = df.filter(pl.col("Station_ID") == station_id).sort("DateTime_UTC0")
        station = station.with_columns(
            (pl.col("DateTime_UTC0").diff() != pl.duration(minutes=10)).fill_null(True)
            .alias("gap"),
            (pl.col("DateTime_UTC0").dt.truncate("1mo")
             != pl.col("DateTime_UTC0").dt.truncate("1mo").shift(1))
            .fill_null(True).alias("new_month"),
        ).with_columns((pl.col("gap") | pl.col("new_month")).cum_sum().alias("segment_id"))
        station_counts = {name: [0, 0] for name in split_records}
        for part in station.partition_by("segment_id"):
            n = part.height
            if n < LOOKBACK + HORIZON:
                continue
            features = part.select(feature_names).to_numpy().astype(np.float32)
            visibility = part["Visibility_km"].to_numpy().astype(np.float32)
            month = part["DateTime_UTC0"][0].month
            invalid = np.concatenate(([0], np.cumsum(~np.isfinite(features).all(axis=1))))
            starts = np.arange(n - LOOKBACK - HORIZON + 1)
            # 輸入截止於 current；target 比 current 晚 HORIZON 個 10 分鐘時段。
            targets = starts + LOOKBACK + HORIZON - 1
            current = starts + LOOKBACK - 1
            valid = ((invalid[starts + LOOKBACK] - invalid[starts]) == 0)
            valid &= np.isfinite(visibility[targets]) & np.isfinite(visibility[current])
            if not valid.any():
                continue
            segment_index = len(segments)
            segments.append({"station": station_id, "features": features, "visibility": visibility,
                             "month": month})
            # 留出站只取訓練月份測試，避免同時更換測站與月份。
            if station_id == SPATIAL_TEST_STATION:
                split = "spatial_test" if month not in VALIDATION_MONTHS | TEMPORAL_TEST_MONTHS else None
            elif month in VALIDATION_MONTHS:
                split = "val"
            elif month in TEMPORAL_TEST_MONTHS:
                split = "temporal_test"
            else:
                split = "train"
            if split is None:
                continue
            selected_starts, selected_targets = starts[valid], targets[valid]
            split_records[split].extend((segment_index, int(a), int(b))
                                        for a, b in zip(selected_starts, selected_targets))
            station_counts[split][0] += len(selected_starts)
            station_counts[split][1] += int((visibility[selected_targets] < FOG_THRESHOLD_KM).sum())
        print(f"{station_id}: " + ", ".join(
            f"{name}={count}/{fog}霧" for name, (count, fog) in station_counts.items() if count))

    if any(not records for records in split_records.values()):
        raise ValueError("至少一個切分沒有有效序列，請調整測站或月份")

    # Use only rows from training windows for scaling; repeated rows are counted once.
    training_rows = {}
    for segment_id, start, _ in split_records["train"]:
        mask = training_rows.setdefault(segment_id, np.zeros(len(segments[segment_id]["features"]), dtype=bool))
        mask[start:start + LOOKBACK] = True
    chunks = [segments[s]["features"][mask] for s, mask in training_rows.items()]
    train_rows = np.concatenate(chunks)
    mean = train_rows.mean(axis=0).astype(np.float32)
    std = train_rows.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0

    datasets = {name: SequenceDataset(segments, records, mean, std)
                for name, records in split_records.items()}
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset)} sequences, {int(dataset.labels.sum())} fog targets, "
              f"{len(np.unique(dataset.station_ids))} stations")
    train_stations = set(datasets["train"].station_ids)
    assert SPATIAL_TEST_STATION not in train_stations
    assert set(datasets["spatial_test"].station_ids) == {SPATIAL_TEST_STATION}
    return datasets, feature_names


if __name__ == "__main__":
    datasets, feature_names = build_datasets("SAT")
    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(train_loader))
    print("第一個 batch:", x.shape, y.shape, feature_names)
