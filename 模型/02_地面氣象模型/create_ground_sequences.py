"""Build ground-weather sequences for the standalone ground GRU model.

Each visibility station receives one fixed-width weather vector per timestamp.
Observations from retained CWA stations within 10 km are aggregated with
normalized exponential distance-decay weights, exp(-distance_km / 5 km).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parents[1]
SOURCE_DATA_DIR = PROJECT_DIR / "來源資料"
DELIVERY_DIR = SOURCE_DATA_DIR / "第一階段可交付成果_2023_地面氣象資料from柯"
DELIVERY_DATA_DIR = DELIVERY_DIR / "資料"
VISIBILITY_PATH = SOURCE_DATA_DIR / "visibility_satellite_2023_btd.parquet"
WEATHER_PATH = DELIVERY_DATA_DIR / "氣象逐時_2023.parquet"
PAIR_PATH = DELIVERY_DATA_DIR / "10公里氣象站配對.csv"
QUALITY_PATH = DELIVERY_DATA_DIR / "氣象測站品質.csv"

DATA_SOURCE = "ACOS"
SPATIAL_TEST_STATION = "C48"
LOOKBACK = 18
HORIZON = 6
FOG_THRESHOLD_KM = 1.0
DISTANCE_SIGMA_KM = 5.0
VALIDATION_MONTHS = {3, 9}
TEMPORAL_TEST_MONTHS = {6, 12}
BATCH_SIZE = 128

CORE_WEATHER_COLUMNS = ["TX01", "RH01", "PS01", "WD01", "WD02", "PP01"]
SATELLITE_AVAILABILITY_COLUMNS = [f"B{i:02d}" for i in range(1, 17)] + [
    "B07_minus_B14", "B07_minus_B13", "B11_minus_B15",
    "B13_minus_B15", "B14_minus_B15", "B15_minus_B16",
]
AGGREGATE_COLUMNS = ["TX01", "RH01", "PS01", "WD01", "PP01"]
TIME_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
GEO_FEATURES = ["Latitude", "Longitude", "Elevation_m"]
SPATIAL_FEATURES = [
    "neighbor_delta_lat_wmean",
    "neighbor_delta_lon_wmean",
    "neighbor_delta_elevation_wmean",
    "nearest_weather_distance_km",
    "paired_weather_station_count",
]

WEATHER_VALUE_FEATURES = [
    feature
    for name in AGGREGATE_COLUMNS
    for feature in (f"{name}_wmean", f"{name}_max")
] + [
    "wind_direction_sin_wmean",
    "wind_direction_cos_wmean",
    "PP01_trace_fraction",
    "PP01_trace_any",
]
WEATHER_COUNT_FEATURES = [f"{name}_count" for name in AGGREGATE_COLUMNS] + [
    "WD02_count",
    "PP01_trace_count",
    "weather_station_rows",
]
MISSING_FEATURES = [f"{name}_missing" for name in WEATHER_VALUE_FEATURES]
GROUND_FEATURES = (
    WEATHER_VALUE_FEATURES
    + WEATHER_COUNT_FEATURES
    + ["weather_age_minutes"]
    + MISSING_FEATURES
    + SPATIAL_FEATURES
    + TIME_FEATURES
    + GEO_FEATURES
)


class GroundSequenceDataset(Dataset):
    def __init__(self, segments, records, mean, std):
        self.segments = segments
        self.records = records
        self.mean = mean
        self.std = std
        self.labels = np.asarray(
            [segments[s]["visibility"][target] < FOG_THRESHOLD_KM for s, _, target in records],
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
        # Missing weather values are imputed with training means. Explicit
        # missing flags and valid-station counts remain in the input.
        filled = np.where(np.isfinite(window), window, self.mean)
        x = torch.from_numpy(((filled - self.mean) / self.std).astype(np.float32))
        y = torch.tensor([self.labels[index]], dtype=torch.float32)
        return x, y


def _weighted_mean_expression(column, alias):
    valid_weight = pl.when(pl.col(column).is_not_null()).then(
        pl.col("_spatial_weight")
    ).otherwise(0.0)
    denominator = valid_weight.sum()
    numerator = (pl.col(column) * pl.col("_spatial_weight")).sum()
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None).alias(alias)


def _load_retained_pairs(visibility_station_ids):
    quality = pd.read_csv(QUALITY_PATH, dtype={"cwa_id": "string"})
    retained_ids = set(quality.loc[quality["station_decision"].eq("保留"), "cwa_id"])
    if len(retained_ids) != 354:
        raise ValueError(f"預期 354 個保留氣象站，實際為 {len(retained_ids)}")

    pairs = pd.read_csv(PAIR_PATH, dtype={"vis_id": "string", "cwa_id": "string"})
    pairs = pairs[
        pairs["vis_id"].isin(visibility_station_ids) & pairs["cwa_id"].isin(retained_ids)
    ].copy()
    if pairs.duplicated(["vis_id", "cwa_id"]).any():
        raise ValueError("能見度站與氣象站配對有重複鍵")
    if pairs["distance_km"].gt(10.0).any() or pairs["distance_km"].le(0.0).any():
        raise ValueError("配對距離必須大於 0 且不超過 10 公里")
    missing_visibility_stations = set(visibility_station_ids) - set(pairs["vis_id"])
    if missing_visibility_stations:
        raise ValueError(f"下列能見度站沒有保留的 10 公里鄰站：{sorted(missing_visibility_stations)}")
    pairs["_spatial_weight"] = np.exp(-pairs["distance_km"] / DISTANCE_SIGMA_KM)
    return pl.from_pandas(pairs)


def _build_ground_weather_frame(visibility):
    station_geo = visibility.select(
        "Station_ID", "Latitude", "Longitude", "Elevation_m"
    ).unique()
    visibility_station_ids = station_geo["Station_ID"].to_list()
    pairs = _load_retained_pairs(visibility_station_ids).join(
        station_geo, left_on="vis_id", right_on="Station_ID", how="inner"
    )

    weight_sum = pl.col("_spatial_weight").sum()
    static_features = pairs.group_by("vis_id").agg(
        (
            ((pl.col("cwa_lat") - pl.col("Latitude")) * pl.col("_spatial_weight")).sum()
            / weight_sum
        ).alias("neighbor_delta_lat_wmean"),
        (
            ((pl.col("cwa_lon") - pl.col("Longitude")) * pl.col("_spatial_weight")).sum()
            / weight_sum
        ).alias("neighbor_delta_lon_wmean"),
        (
            ((pl.col("elevation_m") - pl.col("Elevation_m")) * pl.col("_spatial_weight")).sum()
            / weight_sum
        ).alias("neighbor_delta_elevation_wmean"),
        pl.col("distance_km").min().alias("nearest_weather_distance_km"),
        pl.len().cast(pl.Float64).alias("paired_weather_station_count"),
    )

    weather_station_ids = pairs["cwa_id"].unique().to_list()
    weather = pl.read_parquet(
        WEATHER_PATH,
        columns=[
            "cwa_id", "obs_time_utc", *CORE_WEATHER_COLUMNS, "PP01_trace_flag"
        ],
    ).filter(pl.col("cwa_id").is_in(weather_station_ids))
    if weather.select(pl.struct(["cwa_id", "obs_time_utc"]).is_duplicated().any()).item():
        raise ValueError("氣象逐時資料存在重複的測站—時間鍵")
    weather = weather.with_columns(
        (pl.col("WD02") * np.pi / 180.0).sin().alias("_wind_direction_sin"),
        (pl.col("WD02") * np.pi / 180.0).cos().alias("_wind_direction_cos"),
        pl.col("PP01_trace_flag").cast(pl.Float64).alias("_PP01_trace"),
    )

    joined = weather.join(
        pairs.select("vis_id", "cwa_id", "_spatial_weight"),
        on="cwa_id",
        how="inner",
    )
    aggregate_expressions = []
    for name in AGGREGATE_COLUMNS:
        aggregate_expressions.extend([
            _weighted_mean_expression(name, f"{name}_wmean"),
            pl.col(name).max().alias(f"{name}_max"),
            pl.col(name).is_not_null().sum().cast(pl.Float64).alias(f"{name}_count"),
        ])
    aggregate_expressions.extend([
        _weighted_mean_expression("_wind_direction_sin", "wind_direction_sin_wmean"),
        _weighted_mean_expression("_wind_direction_cos", "wind_direction_cos_wmean"),
        pl.col("WD02").is_not_null().sum().cast(pl.Float64).alias("WD02_count"),
        _weighted_mean_expression("_PP01_trace", "PP01_trace_fraction"),
        pl.col("_PP01_trace").max().alias("PP01_trace_any"),
        pl.col("_PP01_trace").is_not_null().sum().cast(pl.Float64).alias("PP01_trace_count"),
        pl.col("cwa_id").n_unique().cast(pl.Float64).alias("weather_station_rows"),
    ])
    hourly = joined.group_by("vis_id", "obs_time_utc").agg(aggregate_expressions)

    value_missing_expressions = [
        (pl.col(name).is_null() | pl.col(name).is_nan()).cast(pl.Float64).alias(f"{name}_missing")
        for name in WEATHER_VALUE_FEATURES
    ]
    return hourly.join(static_features, on="vis_id", how="left").with_columns(
        value_missing_expressions
    ), pairs


def build_ground_datasets():
    required_paths = [VISIBILITY_PATH, WEATHER_PATH, PAIR_PATH, QUALITY_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"缺少地面模型資料：{missing_paths}")

    visibility = pl.read_parquet(
        VISIBILITY_PATH,
        columns=[
            "Station_ID", "Data_Source", "DateTime_UTC0", "Visibility_km",
            "Latitude", "Longitude", "Elevation_m", *SATELLITE_AVAILABILITY_COLUMNS,
        ],
    ).filter(pl.col("Data_Source") == DATA_SOURCE).with_columns(
        pl.all_horizontal(
            [pl.col(name).is_finite() for name in SATELLITE_AVAILABILITY_COLUMNS]
        ).alias("_comparison_row_available")
    ).drop(SATELLITE_AVAILABILITY_COLUMNS)
    station_ids = sorted(visibility["Station_ID"].unique().to_list())
    if SPATIAL_TEST_STATION not in station_ids or len(station_ids) < 2:
        raise ValueError("空間測試站不存在，或可用測站不足兩站")

    hourly, pairs = _build_ground_weather_frame(visibility)
    local_time = pl.col("DateTime_UTC0") + pl.duration(hours=8)
    hour_angle = (local_time.dt.hour() + local_time.dt.minute() / 60) * (2 * np.pi / 24)
    month_angle = (local_time.dt.month() - 1) * (2 * np.pi / 12)
    visibility = visibility.with_columns(
        pl.col("DateTime_UTC0").dt.truncate("1h").alias("weather_hour_utc"),
        pl.col("DateTime_UTC0").dt.minute().cast(pl.Float64).alias("weather_age_minutes"),
        hour_angle.sin().alias("hour_sin"),
        hour_angle.cos().alias("hour_cos"),
        month_angle.sin().alias("month_sin"),
        month_angle.cos().alias("month_cos"),
    ).join(
        hourly,
        left_on=["Station_ID", "weather_hour_utc"],
        right_on=["vis_id", "obs_time_utc"],
        how="left",
    )

    visibility = visibility.with_columns(
        [pl.col(name).fill_null(0.0).alias(name) for name in WEATHER_COUNT_FEATURES]
        + [pl.col(name).fill_null(1.0).alias(name) for name in MISSING_FEATURES]
    )
    if visibility.select(pl.struct(["Station_ID", "DateTime_UTC0"]).is_duplicated().any()).item():
        raise ValueError("地面特徵合併後產生重複的測站—時間鍵")

    print(
        f"GROUND features: {len(GROUND_FEATURES)}；資料來源：{DATA_SOURCE}；"
        f"能見度站：{len(station_ids)}；保留站配對：{pairs.height}；"
        f"氣象站：{pairs['cwa_id'].n_unique()}；距離權重：exp(-d/{DISTANCE_SIGMA_KM:g})"
    )

    segments = []
    split_records = {name: [] for name in ("train", "val", "temporal_test", "spatial_test")}
    for station_id in station_ids:
        station = visibility.filter(pl.col("Station_ID") == station_id).sort("DateTime_UTC0")
        station = station.with_columns(
            (pl.col("DateTime_UTC0").diff() != pl.duration(minutes=10)).fill_null(True).alias("gap"),
            (
                pl.col("DateTime_UTC0").dt.truncate("1mo")
                != pl.col("DateTime_UTC0").dt.truncate("1mo").shift(1)
            ).fill_null(True).alias("new_month"),
        ).with_columns((pl.col("gap") | pl.col("new_month")).cum_sum().alias("segment_id"))
        station_counts = {name: [0, 0] for name in split_records}
        for part in station.partition_by("segment_id"):
            n = part.height
            if n < LOOKBACK + HORIZON:
                continue
            features = part.select(GROUND_FEATURES).to_numpy().astype(np.float32)
            visibility_values = part["Visibility_km"].to_numpy().astype(np.float32)
            month = part["DateTime_UTC0"][0].month
            starts = np.arange(n - LOOKBACK - HORIZON + 1)
            targets = starts + LOOKBACK + HORIZON - 1
            current = starts + LOOKBACK - 1
            comparison_available = part["_comparison_row_available"].to_numpy()
            invalid = np.concatenate(([0], np.cumsum(~comparison_available)))
            valid = np.isfinite(visibility_values[targets]) & np.isfinite(visibility_values[current])
            # Use the same sequence cohort as the satellite model without
            # feeding any satellite value into this ground-only model.
            valid &= (invalid[starts + LOOKBACK] - invalid[starts]) == 0
            if not valid.any():
                continue
            segment_index = len(segments)
            segments.append({
                "station": station_id,
                "features": features,
                "visibility": visibility_values,
                "month": month,
            })
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
            split_records[split].extend(
                (segment_index, int(start), int(target))
                for start, target in zip(selected_starts, selected_targets)
            )
            station_counts[split][0] += len(selected_starts)
            station_counts[split][1] += int(
                (visibility_values[selected_targets] < FOG_THRESHOLD_KM).sum()
            )
        print(
            f"{station_id}: "
            + ", ".join(
                f"{name}={count}/{fog}霧"
                for name, (count, fog) in station_counts.items()
                if count
            )
        )

    if any(not records for records in split_records.values()):
        raise ValueError("至少一個切分沒有有效序列，請調整測站或月份")

    training_rows = {}
    for segment_id, start, _ in split_records["train"]:
        mask = training_rows.setdefault(
            segment_id, np.zeros(len(segments[segment_id]["features"]), dtype=bool)
        )
        mask[start:start + LOOKBACK] = True
    train_rows = np.concatenate([
        segments[segment_id]["features"][mask]
        for segment_id, mask in training_rows.items()
    ])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(train_rows, axis=0).astype(np.float32)
        std = np.nanstd(train_rows, axis=0).astype(np.float32)
    invalid_statistics = ~np.isfinite(mean) | ~np.isfinite(std)
    if invalid_statistics.any():
        invalid_names = np.asarray(GROUND_FEATURES)[invalid_statistics].tolist()
        raise ValueError(f"訓練資料無法計算下列特徵的標準化統計量：{invalid_names}")
    std[std == 0] = 1.0

    datasets = {
        name: GroundSequenceDataset(segments, records, mean, std)
        for name, records in split_records.items()
    }
    for name, dataset in datasets.items():
        print(
            f"{name}: {len(dataset)} sequences, {int(dataset.labels.sum())} fog targets, "
            f"{len(np.unique(dataset.station_ids))} stations"
        )
    assert SPATIAL_TEST_STATION not in set(datasets["train"].station_ids)
    assert set(datasets["spatial_test"].station_ids) == {SPATIAL_TEST_STATION}
    return datasets, GROUND_FEATURES


if __name__ == "__main__":
    datasets, feature_names = build_ground_datasets()
    loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    print("第一個 batch:", x.shape, y.shape)
    print("特徵:", feature_names)
