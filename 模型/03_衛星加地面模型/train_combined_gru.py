"""Train the standalone satellite-plus-ground-weather GRU fog classifier."""

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from create_combined_sequences import (
    BATCH_SIZE,
    DATA_SOURCE,
    DISTANCE_SIGMA_KM,
    FOG_THRESHOLD_KM,
    HORIZON,
    LOOKBACK,
    PROJECT_DIR,
    SPATIAL_TEST_STATION,
    TEMPORAL_TEST_MONTHS,
    VALIDATION_MONTHS,
    VISIBILITY_PATH,
    WEATHER_PATH,
    build_combined_datasets,
)


MODEL_DIR = Path(__file__).resolve().parent
FOCAL_ALPHA = 0.93 #正類（霧）權重；可手動調整，必須介於 0 與 1
FOCAL_GAMMA = 1.0
HIDDEN_SIZE = 64
DENSE_SIZE = 32
DENSE_DROPOUT = 0.2
EARLY_STOPPING_PATIENCE = 5
THRESHOLD_GRID = np.linspace(0.0, 1.0, 101)


class CombinedSatelliteGroundGRU(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, DENSE_SIZE)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DENSE_DROPOUT)
        self.output = nn.Linear(DENSE_SIZE, 1)

    def forward(self, x):
        last_hidden = self.gru(x)[1][-1]
        dense_output = self.relu(self.dense(last_hidden))
        regularized_output = self.dropout(dense_output)
        return self.output(regularized_output)


class BinaryFocalLossWithLogits(nn.Module):
    """Binary focal loss where alpha is the positive-class weight."""

    def __init__(self, alpha, gamma=1.0):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("Focal loss alpha 必須介於 0 與 1 之間")
        if gamma < 0.0:
            raise ValueError("Focal loss gamma 不可小於 0")
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probabilities = torch.sigmoid(logits)
        p_t = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * (1.0 - p_t).pow(self.gamma) * bce).mean()


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss_sum += criterion(model(x), y).item() * len(x)
            count += len(x)
    if count == 0:
        raise ValueError("評估資料集為空")
    return loss_sum / count


def classification_metrics(actual, predicted):
    tp = int(np.sum(predicted & (actual == 1)))
    fp = int(np.sum(predicted & (actual == 0)))
    fn = int(np.sum(~predicted & (actual == 1)))
    tn = int(np.sum(~predicted & (actual == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    pod = tp / (tp + fn) if tp + fn else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return tp, fp, fn, tn, precision, pod, csi


def print_metrics(name, actual, predicted):
    tp, fp, fn, tn, precision, pod, csi = classification_metrics(actual, predicted)
    print(f"\n=== {name} ===")
    print(f"霧樣本：{int(actual.sum())} / {len(actual)}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Precision: {precision:.4f}, POD/Recall: {pod:.4f}, CSI: {csi:.4f}")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "pod": pod,
        "csi": csi,
    }


def predict_probabilities(model, loader, device):
    logits, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(device)).cpu())
            targets.append(y)
    probabilities = torch.sigmoid(torch.cat(logits)).numpy().ravel()
    actual = torch.cat(targets).numpy().ravel().astype(np.int32)
    return probabilities, actual


def choose_threshold(actual, probabilities):
    best_score, best_threshold = -1.0, 0.5
    for threshold in np.arange(0.05, 0.951, 0.01):
        tp, fp, fn, _, _, _, _ = classification_metrics(actual, probabilities >= threshold)
        score = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score, best_threshold = score, float(threshold)
    return best_threshold, best_score


def save_spatial_threshold_analysis(
    output_dir, file_prefix, model_label, actual, probabilities, selected_threshold
):
    rows = []
    for threshold in THRESHOLD_GRID:
        tp, fp, fn, tn, precision, pod, csi = classification_metrics(
            actual, probabilities >= threshold
        )
        rows.append({
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": pod,
            "csi": csi,
        })

    csv_path = output_dir / f"{file_prefix}_spatial_threshold_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for metric, ylabel in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("csi", "CSI"),
    ):
        plt.figure(figsize=(10, 6))
        plt.plot(
            [row["threshold"] for row in rows],
            [row[metric] for row in rows],
            linewidth=2,
        )
        plt.axvline(0.5, color="gray", linestyle=":", label="Fixed threshold = 0.50")
        plt.axvline(
            selected_threshold,
            color="red",
            linestyle="--",
            label=f"Validation F1 threshold = {selected_threshold:.2f}",
        )
        plt.xlabel("Classification Threshold")
        plt.ylabel(ylabel)
        plt.title(f"{model_label} C48 Spatial Test: {ylabel} by Threshold")
        plt.xlim(0.0, 1.0)
        y_max = min(1.0, max(0.1, max(row[metric] for row in rows) * 1.1))
        plt.ylim(0.0, y_max)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{file_prefix}_spatial_threshold_{metric}.png",
            dpi=300,
        )
        plt.close()

    print(f"C48 門檻明細與三張曲線已寫入：{output_dir}")


def run_experiment(epochs, device, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    datasets, features, preprocessing = build_combined_datasets()
    train_loader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(datasets["val"], batch_size=BATCH_SIZE)
    temporal_loader = DataLoader(datasets["temporal_test"], batch_size=BATCH_SIZE)
    spatial_loader = DataLoader(datasets["spatial_test"], batch_size=BATCH_SIZE)

    fog_count = int(train_loader.dataset.labels.sum())
    non_fog_count = len(train_loader.dataset) - fog_count
    if fog_count == 0 or non_fog_count == 0:
        raise ValueError("訓練集須同時包含霧與非霧樣本")
    criterion = BinaryFocalLossWithLogits(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    # alpha/(1-alpha) 是正負類成本比；Focal Loss 不再另外使用 pos_weight。
    positive_class_cost_ratio = FOCAL_ALPHA / (1.0 - FOCAL_ALPHA)
    print(
        f"Train 霧樣本：{fog_count} / {len(train_loader.dataset)}；"
        f"Focal Loss alpha：{FOCAL_ALPHA:.4f}；gamma：{FOCAL_GAMMA:g}；"
        f"正負類成本比：{positive_class_cost_ratio:.2f}:1"
    )

    model = CombinedSatelliteGroundGRU(len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    train_losses, val_losses = [], []
    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(x)
        train_loss = loss_sum / len(train_loader.dataset)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"Epoch {epoch:02d}/{epochs} | Train Focal: {train_loss:.4f} | "
            f"Val Focal: {val_loss:.4f}"
        )
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping：Validation Focal Loss 已連續 {EARLY_STOPPING_PATIENCE} "
                f"個 epoch 未改善；停止於 epoch {epoch}"
            )
            break

    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "best_combined_satellite_ground_gru_model.pt")
    print(f"Best Epoch: {best_epoch}; Best Validation Focal Loss: {best_val_loss:.4f}")

    val_probabilities, val_actual = predict_probabilities(model, val_loader, device)
    threshold, val_f1 = choose_threshold(val_actual, val_probabilities)
    print(f"Validation F1 最佳門檻：{threshold:.2f}（F1={val_f1:.4f}）")
    print_metrics("Combined GRU Validation @ 0.50", val_actual, val_probabilities >= 0.5)
    print_metrics(
        "Combined GRU Validation @ selected threshold",
        val_actual,
        val_probabilities >= threshold,
    )

    temporal_loss = evaluate_loss(model, temporal_loader, criterion, device)
    temporal_probabilities, temporal_actual = predict_probabilities(
        model, temporal_loader, device
    )
    print(f"Temporal Test Focal Loss: {temporal_loss:.4f}")
    print_metrics(
        "Combined GRU Temporal Test @ 0.50",
        temporal_actual,
        temporal_probabilities >= 0.5,
    )
    temporal_metrics = print_metrics(
        "Combined GRU Temporal Test",
        temporal_actual,
        temporal_probabilities >= threshold,
    )
    print_metrics(
        "Persistence Temporal Test",
        temporal_actual,
        datasets["temporal_test"].persistence < FOG_THRESHOLD_KM,
    )

    spatial_loss = evaluate_loss(model, spatial_loader, criterion, device)
    spatial_probabilities, spatial_actual = predict_probabilities(model, spatial_loader, device)
    print(f"Spatial Test Focal Loss: {spatial_loss:.4f}")
    print_metrics(
        "Combined GRU Spatial Test @ 0.50",
        spatial_actual,
        spatial_probabilities >= 0.5,
    )
    spatial_metrics = print_metrics(
        "Combined GRU Spatial Test",
        spatial_actual,
        spatial_probabilities >= threshold,
    )
    print_metrics(
        "Persistence Spatial Test",
        spatial_actual,
        datasets["spatial_test"].persistence < FOG_THRESHOLD_KM,
    )
    save_spatial_threshold_analysis(
        output_dir,
        "combined",
        "Combined GRU",
        spatial_actual,
        spatial_probabilities,
        threshold,
    )

    metadata = {
        "model": "CombinedSatelliteGroundGRU",
        "features": features,
        "satellite_feature_count": len(preprocessing["satellite_features"]),
        "ground_feature_count": len(preprocessing["ground_features"]),
        "preprocessing": preprocessing,
        "visibility_path": str(VISIBILITY_PATH.relative_to(PROJECT_DIR)),
        "weather_path": str(WEATHER_PATH.relative_to(PROJECT_DIR)),
        "data_source": DATA_SOURCE,
        "spatial_weight": "normalized exp(-distance_km / sigma_km) per feature and timestamp",
        "distance_sigma_km": DISTANCE_SIGMA_KM,
        "threshold": threshold,
        "threshold_selection_metric": "F1",
        "threshold_selection_score": val_f1,
        "threshold_search": {"min": 0.05, "max": 0.95, "step": 0.01},
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "fog_threshold_km": FOG_THRESHOLD_KM,
        "hidden_size": HIDDEN_SIZE,
        "dense_size": DENSE_SIZE,
        "dense_activation": "ReLU",
        "dense_dropout": DENSE_DROPOUT,
        "num_layers": 1,
        "seed": 42,
        "spatial_test_station": SPATIAL_TEST_STATION,
        "validation_months": sorted(VALIDATION_MONTHS),
        "temporal_test_months": sorted(TEMPORAL_TEST_MONTHS),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "completed_epochs": len(train_losses),
        "loss_function": "BinaryFocalLossWithLogits",
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "positive_class_cost_ratio": positive_class_cost_ratio,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with (output_dir / "combined_gru_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    plt.figure(figsize=(10, 6))
    completed_epochs = range(1, len(train_losses) + 1)
    plt.plot(completed_epochs, train_losses, label="Training Loss")
    plt.plot(completed_epochs, val_losses, label="Validation Loss")
    plt.axvline(best_epoch, linestyle="--", label=f"Best Epoch = {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Focal Loss")
    plt.title("Satellite + Ground-weather GRU Focal Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_validation_loss_combined.png", dpi=300)
    plt.close()

    return [
        {
            "mode": "SAT_GROUND",
            "split": "temporal_test",
            "features": len(features),
            "best_epoch": best_epoch,
            "threshold": threshold,
            **temporal_metrics,
        },
        {
            "mode": "SAT_GROUND",
            "split": "spatial_test",
            "features": len(features),
            "best_epoch": best_epoch,
            "threshold": threshold,
            **spatial_metrics,
        },
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="衛星加地面氣象 GRU 霧偵測模型")
    parser.add_argument("--epochs", type=int, default=30, help="最多訓練 epoch 數")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR / "results",
        help="模型與評估輸出目錄",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs 至少須為 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    rows = run_experiment(args.epochs, device, args.output_dir)
    output_path = args.output_dir / "combined_results.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n=== Results（已寫入 {output_path}）===")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
