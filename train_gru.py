"""Train one GRU on independent sequences from multiple stations."""

import copy
import csv
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from create_sequences import (
    BATCH_SIZE, FEATURE_SETS, FOG_THRESHOLD_KM, build_datasets,
)

POS_WEIGHT_SCALE = 0.5  # 1.0 為原本的反類別比例權重；調低可減少漏報懲罰
POS_WEIGHT_CAP = 5.0  # 多站霧比例更低，避免正類權重暴增而大量誤報


class VisibilityGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc(self.gru(x)[1][-1])  # logits; sigmoid only for probabilities


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
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "pod": pod, "csi": csi}


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
    # F0.5 比 F1 更重視 precision，減少假陽性；只用 validation 選門檻。
    best_score, best_threshold = -1.0, 0.5
    for threshold in np.arange(0.05, 0.951, 0.01):
        tp, fp, fn, _, _, _, _ = classification_metrics(actual, probabilities >= threshold)
        score = 1.25 * tp / (1.25 * tp + fp + 0.25 * fn) if tp else 0.0
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score, best_threshold = score, float(threshold)
    return best_threshold, best_score


def run_experiment(feature_mode, epochs, device):
    print(f"\n{'=' * 20} {feature_mode} {'=' * 20}")
    torch.manual_seed(42)
    np.random.seed(42)
    datasets, features = build_datasets(feature_mode)
    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(datasets["val"], batch_size=BATCH_SIZE)
    test_loader = DataLoader(datasets["temporal_test"], batch_size=BATCH_SIZE)
    spatial_test_loader = DataLoader(datasets["spatial_test"], batch_size=BATCH_SIZE)
    fog_count = int(train_loader.dataset.labels.sum())
    non_fog_count = len(train_loader.dataset) - fog_count
    if fog_count == 0 or non_fog_count == 0:
        raise ValueError("訓練集須同時包含霧與非霧樣本")
    # 多站霧比例很低；權重設上限，避免一味偏向報霧而產生大量 FP。
    pos_weight = torch.tensor(
        [min(POS_WEIGHT_SCALE * non_fog_count / fog_count, POS_WEIGHT_CAP)], device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Train 霧樣本：{fog_count} / {len(train_loader.dataset)}；pos_weight：{pos_weight.item():.2f}")

    model = VisibilityGRU(len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses = [], []
    best_val_loss, best_epoch, best_state = float("inf"), 0, None
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
        print(f"Epoch {epoch:02d}/{epochs} | Train BCE: {train_loss:.4f} | Val BCE: {val_loss:.4f}")

    model.load_state_dict(best_state)
    torch.save(best_state, f"best_gru_fog_{feature_mode.lower()}_model.pt")
    print(f"Best Epoch: {best_epoch}; Best Validation weighted BCE: {best_val_loss:.4f}")
    val_probabilities, val_actual = predict_probabilities(model, val_loader, device)
    # 門檻只從 validation 選，temporal/spatial test 都不能參與調整。
    threshold, val_f05 = choose_threshold(val_actual, val_probabilities)
    print(f"Validation F0.5 最佳門檻：{threshold:.2f}（F0.5={val_f05:.4f}）")
    print_metrics("GRU Validation @ 0.50", val_actual, val_probabilities >= 0.5)
    print_metrics("GRU Validation @ selected threshold", val_actual, val_probabilities >= threshold)

    test_loss = evaluate_loss(model, test_loader, criterion, device)
    probabilities, actual = predict_probabilities(model, test_loader, device)
    print(f"Test weighted BCE: {test_loss:.4f}")
    print_metrics("GRU Test @ 0.50", actual, probabilities >= 0.5)
    temporal_metrics = print_metrics("GRU Test", actual, probabilities >= threshold)
    print_metrics("Persistence Test", actual,
                  datasets["temporal_test"].persistence < FOG_THRESHOLD_KM)

    spatial_loss = evaluate_loss(model, spatial_test_loader, criterion, device)
    spatial_probabilities, spatial_actual = predict_probabilities(model, spatial_test_loader, device)
    print(f"Spatial Test weighted BCE: {spatial_loss:.4f}")
    spatial_metrics = print_metrics("GRU Spatial Test", spatial_actual,
                                    spatial_probabilities >= threshold)
    print_metrics("Persistence Spatial Test", spatial_actual,
                  datasets["spatial_test"].persistence < FOG_THRESHOLD_KM)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.axvline(best_epoch, linestyle="--", label=f"Best Epoch = {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE Loss")
    plt.title("Multi-station GRU Fog Detection Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"training_validation_loss_fog_{feature_mode.lower()}.png", dpi=300)
    plt.close()
    return [
        {"mode": feature_mode, "split": "temporal_test", "features": len(features),
         "best_epoch": best_epoch, "threshold": threshold, **temporal_metrics},
        {"mode": feature_mode, "split": "spatial_test", "features": len(features),
         "best_epoch": best_epoch, "threshold": threshold, **spatial_metrics},
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="多測站 GRU 霧偵測特徵比較")
    parser.add_argument(
        "--mode",
        choices=["ALL", *FEATURE_SETS],
        default="ALL",
        help="選擇單一特徵組合；預設 ALL 依序執行三組",
    )
    parser.add_argument("--epochs", type=int, default=30, help="每組訓練 epoch 數")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs 至少須為 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    modes = list(FEATURE_SETS) if args.mode == "ALL" else [args.mode]
    rows = []
    for mode in modes:
        rows.extend(run_experiment(mode, args.epochs, device))
    output_path = (
        "fog_feature_comparison.csv" if args.mode == "ALL"
        else f"fog_feature_results_{args.mode.lower()}.csv"
    )
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n=== Results（已寫入 {output_path}）===")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
