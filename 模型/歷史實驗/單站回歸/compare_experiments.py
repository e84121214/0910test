"""Compare satellite-only GRU experiments on identical target timestamps.

Run: .venv/Scripts/python compare_experiments.py --station C06 --output comparison_results_C06.csv
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_DIR / "來源資料" / "visibility_satellite_2023_clean.parquet"
FEATURES = [f"B{i:02d}" for i in range(1, 17)]
TEST_MONTHS = {6, 12}
VALIDATION_SCHEMES = {"march_september": {3, 9}, "october": {10}}
SEED = 42


def make_sequences(station_df, minutes, split_month=True):
    if minutes == 30:
        station_df = station_df.filter(
            pl.col("DateTime_UTC0").dt.minute().is_in([0, 30])
        )
    lookback, horizon = (18, 6) if minutes == 10 else (6, 2)
    station_df = station_df.sort("DateTime_UTC0").with_columns(
        pl.col("DateTime_UTC0").diff().alias("time_diff")
    )
    boundary = pl.col("time_diff") != pl.duration(minutes=minutes)
    if split_month:
        boundary = boundary | (
            pl.col("DateTime_UTC0").dt.truncate("1mo")
            != pl.col("DateTime_UTC0").dt.truncate("1mo").shift(1)
        )
    station_df = station_df.with_columns(
        boundary.fill_null(True).cum_sum().alias("segment_id")
    )
    samples = {}
    for segment in station_df.partition_by("segment_id"):
        if segment.height < lookback + horizon:
            continue
        x = segment.select(FEATURES).to_numpy().astype(np.float32)
        y = segment["Visibility_km"].to_numpy().astype(np.float32)
        times = segment["DateTime_UTC0"].to_numpy()
        for start in range(segment.height - lookback - horizon + 1):
            target = start + lookback + horizon - 1
            current = start + lookback - 1
            window = x[start:start + lookback]
            if not (
                np.isfinite(window).all()
                and np.isfinite(y[target])
                and np.isfinite(y[current])
            ):
                continue
            key = np.datetime64(times[target], "ns")
            if key in samples:
                raise ValueError(f"Duplicate target time: {key}")
            samples[key] = (window, y[target], y[current])
    return samples


class VisibilityGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=len(FEATURES), hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc(self.gru(x)[1][-1])


def scores(actual, predicted):
    error = predicted - actual
    return float(np.sqrt(np.mean(error ** 2))), float(np.mean(np.abs(error)))


def run_one(samples, keys, minutes, scheme, val_months, epochs, device,
            test_months=TEST_MONTHS):
    months = np.array([int(str(key)[5:7]) for key in keys])
    test = np.isin(months, list(test_months))
    val = np.isin(months, list(val_months))
    train = ~(test | val)
    if not (train.any() and val.any() and test.any()):
        raise ValueError(f"Empty split: {minutes} min, {scheme}")

    x = np.stack([samples[key][0] for key in keys])
    y = np.array([samples[key][1] for key in keys], dtype=np.float32)
    persistence = np.array([samples[key][2] for key in keys], dtype=np.float32)
    mean = x[train].mean(axis=(0, 1), keepdims=True)
    std = x[train].std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1
    x = (x - mean) / std

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model = VisibilityGRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    rng = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train, None])),
        batch_size=128, shuffle=True, generator=rng,
    )
    val_x = torch.from_numpy(x[val]).to(device)
    val_y = torch.from_numpy(y[val, None]).to(device)
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x), val_y).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: weight.detach().clone() for name, weight in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(x[test]).to(device)).cpu().numpy().ravel()

    rows = []
    for month in (*sorted(test_months), None):
        subset = test if month is None else test & (months == month)
        selected_predicted = predicted if month is None else predicted[months[test] == month]
        gru_rmse, gru_mae = scores(y[subset], selected_predicted)
        base_rmse, base_mae = scores(y[subset], persistence[subset])
        fog = y[subset] < 1
        fog_gru_mae = float(np.mean(np.abs(selected_predicted[fog] - y[subset][fog]))) if fog.any() else ""
        fog_base_mae = float(np.mean(np.abs(persistence[subset][fog] - y[subset][fog]))) if fog.any() else ""
        rows.append({
            "minutes": minutes, "validation": scheme,
            "test_month": month if month is not None else "all",
            "train_n": int(train.sum()), "val_n": int(val.sum()), "test_n": int(subset.sum()),
            "fog_targets": int((y[subset] < 1).sum()), "best_epoch": best_epoch,
            "gru_rmse": gru_rmse, "gru_mae": gru_mae,
            "persistence_rmse": base_rmse, "persistence_mae": base_mae,
            "fog_gru_mae": fog_gru_mae, "fog_persistence_mae": fog_base_mae,
            "predicted_fog_n": int((selected_predicted < 1).sum()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--station", default="C19")
    parser.add_argument("--output", type=Path, default=Path("comparison_results.csv"))
    args = parser.parse_args()
    station_df = pl.read_parquet(DATA_PATH).filter(pl.col("Station_ID") == args.station)
    sequences = {minutes: make_sequences(station_df, minutes) for minutes in (10, 30)}
    keys = sorted(set(sequences[10]) & set(sequences[30]))
    if not keys:
        raise ValueError("No shared valid target timestamps")
    print(f"Station: {args.station}; shared valid target timestamps: {len(keys)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    rows = []
    for minutes in (10, 30):
        print(f"Training {minutes} min / seasonal calendar, all targets", flush=True)
        rows.extend(run_one(
            sequences[minutes], sorted(sequences[minutes]), minutes,
            "seasonal_all_targets", {3, 9}, args.epochs, device,
        ))
    for scheme, val_months in VALIDATION_SCHEMES.items():
        for minutes in (10, 30):
            print(f"Training {minutes} min / validation {scheme}", flush=True)
            rows.extend(run_one(sequences[minutes], keys, minutes, scheme, val_months, args.epochs, device))
    for minutes in (10, 30):
        print(f"Training {minutes} min / original calendar, shared targets", flush=True)
        rows.extend(run_one(
            sequences[minutes], keys, minutes, "october_nov_dec_shared",
            {10}, args.epochs, device, {11, 12},
        ))
    original = make_sequences(station_df, 10, split_month=False)
    print("Training original 10 min / original calendar, all targets", flush=True)
    rows.extend(run_one(
        original, sorted(original), 10, "original_all_targets",
        {10}, args.epochs, device, {11, 12},
    ))
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
