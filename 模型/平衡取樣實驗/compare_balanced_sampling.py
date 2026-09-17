"""Summarize standard, station-balanced, and event-balanced experiments."""

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXPERIMENT_DIR.parent
STRATEGIES = ("standard", "station_balanced", "event_balanced")
MODELS = (
    ("SAT", MODELS_DIR / "01_衛星模型" / "results", "sat", "fog_feature_results_sat.csv"),
    ("GROUND", MODELS_DIR / "02_地面氣象模型" / "results", "ground", "ground_weather_results.csv"),
    (
        "SAT_GROUND",
        MODELS_DIR / "03_衛星加地面模型" / "results",
        "combined",
        "combined_results.csv",
    ),
)


def strategy_dir(root, strategy):
    return root if strategy == "standard" else root / strategy


def read_rows(path):
    with path.open(encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def read_losses(log_path):
    text = log_path.read_text(encoding="utf-8")
    temporal_match = re.search(r"(?:Temporal Test|Test) weighted BCE: ([0-9.]+)", text)
    spatial_match = re.search(r"Spatial Test weighted BCE: ([0-9.]+)", text)
    if not temporal_match or not spatial_match:
        raise ValueError(f"找不到測試 BCE：{log_path}")
    return float(temporal_match.group(1)), float(spatial_match.group(1))


def best_spatial_csi(path):
    rows = read_rows(path)
    best = max(rows, key=lambda row: float(row["csi"]))
    return float(best["threshold"]), float(best["csi"])


def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = []
    for model, root, prefix, result_name in MODELS:
        baseline = {row["split"]: row for row in read_rows(root / result_name)}
        for strategy in STRATEGIES:
            folder = strategy_dir(root, strategy)
            rows = {row["split"]: row for row in read_rows(folder / result_name)}
            temporal_bce, spatial_bce = read_losses(folder / "training.log")
            oracle_threshold, oracle_csi = best_spatial_csi(
                folder / f"{prefix}_spatial_threshold_metrics.csv"
            )
            for split, row in rows.items():
                baseline_row = baseline[split]
                csi = float(row["csi"])
                baseline_csi = float(baseline_row["csi"])
                comparison.append({
                    "model": model,
                    "sampling": strategy,
                    "split": split,
                    "threshold": float(row["threshold"]),
                    "tp": int(row["tp"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "tn": int(row["tn"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["pod"]),
                    "csi": csi,
                    "delta_csi_vs_standard": csi - baseline_csi,
                    "relative_csi_change_pct": (
                        (csi / baseline_csi - 1) * 100 if baseline_csi else ""
                    ),
                    "weighted_bce": temporal_bce if split == "temporal_test" else spatial_bce,
                    "c48_diagnostic_best_threshold": (
                        oracle_threshold if split == "spatial_test" else ""
                    ),
                    "c48_diagnostic_best_csi": (
                        oracle_csi if split == "spatial_test" else ""
                    ),
                })

    output_csv = EXPERIMENT_DIR / "balanced_sampling_comparison.csv"
    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    model_names = [item[0] for item in MODELS]
    strategy_labels = {
        "standard": "Standard",
        "station_balanced": "Station balanced",
        "event_balanced": "Event balanced",
    }
    colors = {"standard": "#4C78A8", "station_balanced": "#F58518", "event_balanced": "#54A24B"}
    x_positions = list(range(len(model_names)))
    width = 0.24
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    chart_specs = (
        ("temporal_test", "csi", "Temporal test CSI"),
        ("spatial_test", "csi", "C48 CSI at validation F1 threshold"),
        ("spatial_test", "c48_diagnostic_best_csi", "C48 diagnostic maximum CSI"),
    )
    for axis, (split, field, title) in zip(axes, chart_specs):
        for offset_index, strategy in enumerate(STRATEGIES):
            values = []
            for model in model_names:
                row = next(
                    item for item in comparison
                    if item["model"] == model
                    and item["sampling"] == strategy
                    and item["split"] == split
                )
                values.append(float(row[field]))
            positions = [x + (offset_index - 1) * width for x in x_positions]
            axis.bar(
                positions,
                values,
                width=width,
                label=strategy_labels[strategy],
                color=colors[strategy],
            )
        axis.set_title(title)
        axis.set_xticks(x_positions, model_names)
        axis.set_ylabel("CSI")
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(EXPERIMENT_DIR / "balanced_sampling_csi_comparison.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
