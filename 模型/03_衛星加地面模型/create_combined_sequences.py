"""Combine aligned satellite and ground-weather sequences for one GRU."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


MODEL_DIR = Path(__file__).resolve().parent
MODELS_DIR = MODEL_DIR.parent
PROJECT_DIR = MODELS_DIR.parent
SATELLITE_SEQUENCE_PATH = MODELS_DIR / "01_衛星模型" / "create_sequences.py"
GROUND_SEQUENCE_PATH = MODELS_DIR / "02_地面氣象模型" / "create_ground_sequences.py"


def _load_sequence_module(module_name, path):
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"無法載入資料序列模組：{path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


satellite_sequences = _load_sequence_module(
    "combined_satellite_sequences", SATELLITE_SEQUENCE_PATH
)
ground_sequences = _load_sequence_module(
    "combined_ground_sequences", GROUND_SEQUENCE_PATH
)

BATCH_SIZE = ground_sequences.BATCH_SIZE
FOG_THRESHOLD_KM = ground_sequences.FOG_THRESHOLD_KM
LOOKBACK = ground_sequences.LOOKBACK
HORIZON = ground_sequences.HORIZON
DATA_SOURCE = ground_sequences.DATA_SOURCE
SPATIAL_TEST_STATION = ground_sequences.SPATIAL_TEST_STATION
VALIDATION_MONTHS = ground_sequences.VALIDATION_MONTHS
TEMPORAL_TEST_MONTHS = ground_sequences.TEMPORAL_TEST_MONTHS
DISTANCE_SIGMA_KM = ground_sequences.DISTANCE_SIGMA_KM
VISIBILITY_PATH = ground_sequences.VISIBILITY_PATH
WEATHER_PATH = ground_sequences.WEATHER_PATH

SATELLITE_INPUT_FEATURES = (
    satellite_sequences.SATELLITE_FEATURES
    + satellite_sequences.BAND_DIFFERENCE_FEATURES
)


class CombinedSequenceDataset(Dataset):
    def __init__(self, satellite_dataset, ground_dataset, satellite_indices):
        if len(satellite_dataset) != len(ground_dataset):
            raise ValueError("衛星與地面資料集長度不同")
        if satellite_dataset.records != ground_dataset.records:
            raise ValueError("衛星與地面序列索引沒有逐筆對齊")
        if not np.array_equal(satellite_dataset.labels, ground_dataset.labels):
            raise ValueError("衛星與地面標籤沒有逐筆對齊")
        if not np.array_equal(satellite_dataset.station_ids, ground_dataset.station_ids):
            raise ValueError("衛星與地面測站順序沒有逐筆對齊")
        if not np.allclose(
            satellite_dataset.persistence,
            ground_dataset.persistence,
            equal_nan=True,
        ):
            raise ValueError("衛星與地面 persistence 基準沒有逐筆對齊")
        self.satellite_dataset = satellite_dataset
        self.ground_dataset = ground_dataset
        self.satellite_indices = satellite_indices
        self.records = ground_dataset.records
        self.labels = ground_dataset.labels
        self.persistence = ground_dataset.persistence
        self.station_ids = ground_dataset.station_ids

    def __len__(self):
        return len(self.ground_dataset)

    def __getitem__(self, index):
        satellite_x, satellite_y = self.satellite_dataset[index]
        ground_x, ground_y = self.ground_dataset[index]
        if not torch.equal(satellite_y, ground_y):
            raise ValueError("衛星與地面 batch 標籤不一致")
        x = torch.cat((satellite_x[:, self.satellite_indices], ground_x), dim=1)
        return x, ground_y


def build_combined_datasets():
    if not SATELLITE_SEQUENCE_PATH.exists() or not GROUND_SEQUENCE_PATH.exists():
        raise FileNotFoundError("找不到第一或第二種模型的資料序列程式")
    if (
        satellite_sequences.BATCH_SIZE != ground_sequences.BATCH_SIZE
        or satellite_sequences.LOOKBACK != ground_sequences.LOOKBACK
        or satellite_sequences.HORIZON != ground_sequences.HORIZON
        or satellite_sequences.FOG_THRESHOLD_KM != ground_sequences.FOG_THRESHOLD_KM
        or satellite_sequences.SPATIAL_TEST_STATION != ground_sequences.SPATIAL_TEST_STATION
        or satellite_sequences.VALIDATION_MONTHS != ground_sequences.VALIDATION_MONTHS
        or satellite_sequences.TEMPORAL_TEST_MONTHS != ground_sequences.TEMPORAL_TEST_MONTHS
    ):
        raise ValueError("第一與第二種模型的序列或切分設定不一致")

    satellite_datasets, satellite_features = satellite_sequences.build_datasets("SAT")
    ground_datasets, ground_features = ground_sequences.build_ground_datasets()
    satellite_indices = [satellite_features.index(name) for name in SATELLITE_INPUT_FEATURES]
    combined_features = SATELLITE_INPUT_FEATURES + ground_features
    if len(combined_features) != len(set(combined_features)):
        raise ValueError("混合模型出現重複特徵名稱")

    datasets = {
        split: CombinedSequenceDataset(
            satellite_datasets[split], ground_datasets[split], satellite_indices
        )
        for split in satellite_datasets
    }
    preprocessing = {
        "satellite_features": SATELLITE_INPUT_FEATURES,
        "satellite_mean": satellite_datasets["train"].mean[satellite_indices].tolist(),
        "satellite_std": satellite_datasets["train"].std[satellite_indices].tolist(),
        "ground_features": ground_features,
        "ground_mean": ground_datasets["train"].mean.tolist(),
        "ground_std": ground_datasets["train"].std.tolist(),
    }
    print(
        f"COMBINED features: {len(combined_features)} "
        f"({len(SATELLITE_INPUT_FEATURES)} satellite + {len(ground_features)} ground)"
    )
    for name, dataset in datasets.items():
        print(
            f"{name}: {len(dataset)} sequences, {int(dataset.labels.sum())} fog targets, "
            f"{len(np.unique(dataset.station_ids))} stations"
        )
    return datasets, combined_features, preprocessing


if __name__ == "__main__":
    datasets, feature_names, _ = build_combined_datasets()
    loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    print("第一個 batch:", x.shape, y.shape)
    print("特徵:", feature_names)
