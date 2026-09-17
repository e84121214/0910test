"""Training-only sampling strategies shared by the three GRU experiments."""

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


SAMPLING_STRATEGIES = ("standard", "station_balanced", "event_balanced")


def _make_sampler(weights, sample_count, seed):
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=sample_count,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _station_class_balanced_weights(dataset):
    """Reduce station imbalance within each class with inverse-square-root weights.

    Total positive and negative sampling mass is kept equal to the original
    class counts. Square-root smoothing limits extreme repetition at stations
    that contain only one or a few fog samples.
    """
    labels = np.asarray(dataset.labels, dtype=np.int8)
    stations = np.asarray(dataset.station_ids)
    weights = np.zeros(len(dataset), dtype=np.float64)
    group_counts = {}

    for label in (0, 1):
        class_mask = labels == label
        class_total = int(class_mask.sum())
        class_stations = np.unique(stations[class_mask])
        if class_total == 0 or len(class_stations) == 0:
            raise ValueError(f"訓練集缺少類別 {label}，無法建立測站平衡取樣")
        class_groups = []
        raw_mass = 0.0
        for station in class_stations:
            group_mask = class_mask & (stations == station)
            group_count = int(group_mask.sum())
            raw_weight = 1.0 / np.sqrt(group_count)
            class_groups.append((group_mask, raw_weight))
            raw_mass += raw_weight * group_count
            group_counts[f"label_{label}:{station}"] = group_count
        normalizer = class_total / raw_mass
        for group_mask, raw_weight in class_groups:
            weights[group_mask] = raw_weight * normalizer

    return weights, {
        "description": "inverse-sqrt station-class balancing; original class mass preserved",
        "positive_station_count": int(len(np.unique(stations[labels == 1]))),
        "negative_station_count": int(len(np.unique(stations[labels == 0]))),
        "station_class_counts": group_counts,
    }


def _positive_event_balanced_weights(dataset):
    """Give every contiguous positive target event equal expected mass.

    A fog event consists of consecutive positive target indices in the same
    uninterrupted station segment. Negative rows keep unit weight. Positive
    weights are normalized to preserve the original total positive mass.
    """
    labels = np.asarray(dataset.labels, dtype=np.int8)
    weights = np.ones(len(dataset), dtype=np.float64)
    positives_by_segment = defaultdict(list)
    for sample_index, ((segment_id, _, target_index), label) in enumerate(
        zip(dataset.records, labels)
    ):
        if label == 1:
            positives_by_segment[int(segment_id)].append(
                (int(target_index), sample_index)
            )

    events = []
    for items in positives_by_segment.values():
        items.sort()
        current_event = []
        previous_target = None
        for target_index, sample_index in items:
            if previous_target is None or target_index == previous_target + 1:
                current_event.append(sample_index)
            else:
                events.append(current_event)
                current_event = [sample_index]
            previous_target = target_index
        if current_event:
            events.append(current_event)

    positive_total = int(labels.sum())
    if positive_total == 0 or not events:
        raise ValueError("訓練集沒有霧事件，無法建立事件平衡取樣")
    event_mass = positive_total / len(events)
    event_lengths = []
    for event in events:
        weights[event] = event_mass / len(event)
        event_lengths.append(len(event))

    return weights, {
        "description": "equal mass per contiguous positive event; original class mass preserved",
        "positive_event_count": len(events),
        "event_length_min": int(min(event_lengths)),
        "event_length_median": float(np.median(event_lengths)),
        "event_length_max": int(max(event_lengths)),
    }


def build_training_sampler(dataset, strategy, seed=42):
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"未知取樣策略：{strategy}")
    if strategy == "standard":
        return None, {"description": "standard shuffled sampling without replacement"}
    if strategy == "station_balanced":
        weights, diagnostics = _station_class_balanced_weights(dataset)
    else:
        weights, diagnostics = _positive_event_balanced_weights(dataset)

    labels = np.asarray(dataset.labels, dtype=np.int8)
    diagnostics.update({
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "positive_sampling_mass": float(weights[labels == 1].sum()),
        "negative_sampling_mass": float(weights[labels == 0].sum()),
    })
    return _make_sampler(weights, len(dataset), seed), diagnostics
