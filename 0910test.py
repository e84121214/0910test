import polars as pl

DATA_PATH = "visibility_satellite_2023_clean.parquet"

df = pl.read_parquet(DATA_PATH)
'''
print("資料 shape：", df.shape)
print("\n欄位：")
print(df.columns)

print("\n資料型態：")
print(df.schema)

print("\n前 5 筆：")
print(df.head())

print("\n測站數：")
print(df["Station_ID"].n_unique())

print("\n時間範圍：")
print(
    df.select(
        pl.col("DateTime_UTC0").min().alias("start"),
        pl.col("DateTime_UTC0").max().alias("end")
    )
)
'''
#算各測站資料筆數
station_counts = (
    df.group_by("Station_ID")
    .agg(
        pl.len().alias("n"),
        pl.col("Visibility_km").null_count().alias("visibility_nulls")
    )
    .sort("n", descending=True)
)


with pl.Config(tbl_rows=20):
    print(station_counts.head(20))

#看C19的每10分鐘時間連續性
STATION = "C19"

station_df = (
    df.filter(pl.col("Station_ID") == STATION)
      .sort("DateTime_UTC0")
      .with_columns(
          pl.col("DateTime_UTC0")
            .diff()
            .alias("time_diff")
      )
)

print(station_df.shape)

print(
    station_df
    .group_by("time_diff")
    .len()
    .sort("len", descending=True)
    .head(20)
)

#當時間差不等於10分鐘，就新開一個segment
station_df = (
    df.filter(pl.col("Station_ID") == STATION)
      .sort("DateTime_UTC0")
      .with_columns(
          pl.col("DateTime_UTC0")
            .diff()
            .alias("time_diff")
      )
)

station_df = station_df.with_columns(
    (pl.col("time_diff") != pl.duration(minutes=10))
    .fill_null(True)
    .cum_sum()
    .alias("segment_id")
)

segment_lengths = (
    station_df
    .group_by("segment_id")
    .agg(
        pl.len().alias("length"),
        pl.col("DateTime_UTC0").min().alias("start"),
        pl.col("DateTime_UTC0").max().alias("end")
    )
    .sort("length", descending=True)
)

with pl.Config(tbl_rows=30):
    print(segment_lengths.head(30))

#計算可用的segment數量
usable_segments = segment_lengths.filter(
    pl.col("length") >= 24
)

print("可使用的連續區段數：", usable_segments.height)

print(
    "這些區段總資料筆數：",
    usable_segments["length"].sum()
)

#預估能產生的GRU sample數
LOOKBACK = 18
HORIZON = 6

usable_segments = usable_segments.with_columns(
    (
        pl.col("length") - LOOKBACK - HORIZON + 1
    ).alias("n_samples")
)

print(
    "預估可產生的 GRU samples：",
    usable_segments["n_samples"].sum()
)

#檢查 C19 在一天 24 小時中的資料分布

hour_counts = (
    station_df
    .with_columns(
        pl.col("DateTime_UTC0")
        .dt.hour()
        .alias("hour")
    )
    .group_by("hour")
    .agg(
        pl.len().alias("n")
    )
    .sort("hour")
)

with pl.Config(tbl_rows=-1):
    print("\n=== 每個 UTC 小時的資料筆數 ===")
    print(hour_counts)