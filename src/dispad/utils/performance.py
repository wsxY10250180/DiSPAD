"""Shared timing protocol used by all three anomaly detection projects."""

import json
import time
from pathlib import Path

import torch


def synchronized_time(device):
    """Wall time after all pending work on the inference device has finished."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def write_performance_stats(output_path, model_load_time_s, categories, configuration):
    """Aggregate actual image counts/times; count the shared backbone only once."""
    categories = [dict(row) for row in categories]
    for row in categories:
        elapsed = row["inference_time_s"]
        row["fps"] = row["num_images"] / elapsed if elapsed > 0 else None
    reference_time = sum(row["reference_init_time_s"] for row in categories)
    inference_time = sum(row["inference_time_s"] for row in categories)
    num_images = sum(row["num_images"] for row in categories)
    stats = {
        "initialization_time_s": model_load_time_s + reference_time,
        "model_load_time_s": model_load_time_s,
        "reference_init_time_s": reference_time,
        "num_images": num_images,
        "inference_time_s": inference_time,
        "fps": num_images / inference_time if inference_time > 0 else None,
        "timing_protocol": {
            "clock": "perf_counter, CUDA synchronized at both boundaries",
            "initialization": "model loading + reference image loading/augmentation/feature extraction + reference model fitting",
            "inference": "decoded RGB image -> preprocessing + feature extraction + scoring + map postprocessing + image score",
            "excluded": "test image I/O, saving, visualization, metrics, validation threshold selection, warmup",
            "warmup": "repeat the first complete test batch before measuring each category",
            "aggregation": "total images / total inference seconds; shared model loading counted once per report",
        },
        "configuration": configuration,
        "categories": categories,
    }
    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    return stats
