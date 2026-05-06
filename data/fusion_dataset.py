"""
Fusion dataset — loads matched (sensor, video) pairs with strict tuple matching.
Applies modality-specific preprocessing and returns ((x_sensor, x_video), label) tuples.
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.sensor_dataset import add_velocity as add_velocity_sensor
from data.sensor_dataset import extract_imu_features
from data.video_dataset import add_velocity as add_velocity_video
from data.video_dataset import extract_wrist_deltas, get_landmark_indices


def extract_matched_keys(sensor_samples, video_samples):
    """Match sensor and video samples on (subject, label) and return flat 3-tuples.

    Args:
        sensor_samples: list of (subject, label, raw_iner)
        video_samples: list of (subject, label, raw_pose)

    Returns:
        matched_samples: list of (subject, label, (sensor_raw, video_raw)) 3-tuples.
        Same format as standalone sensor/video samples, with `data` being a paired tuple.
    """
    sensor_dict = {}
    for s, y, raw in sensor_samples:
        sensor_dict.setdefault((s, y), []).append(raw)

    video_dict = {}
    for s, y, raw in video_samples:
        video_dict.setdefault((s, y), []).append(raw)

    matched_keys = set(sensor_dict.keys()) & set(video_dict.keys())
    matched_samples = []

    for key in sorted(matched_keys):
        s, y = key
        # Pair up by index — for each (subject,label), take min count of trials
        n = min(len(sensor_dict[key]), len(video_dict[key]))
        for i in range(n):
            paired_data = (sensor_dict[key][i], video_dict[key][i])
            matched_samples.append((s, y, paired_data))

    return matched_samples


class FusionDataset(Dataset):
    """Load matched (sensor, video) pairs with per-modality preprocessing.

    Args:
        samples: list of (label, (sensor_raw, video_raw)) tuples
                 where sensor_raw is (T, 6) and video_raw is (T, 66)
        sensor_max_len: pad/truncate sensor sequences to this length
        video_max_len: pad/truncate video sequences to this length
        feature_type: "raw+velocity" or "hand_crafted" for sensor
        landmark_set: landmark subset for video
        normalization_type: "per_sample" or "global"
        global_stats_sensor: (mean, std) tuple for sensor, or None
        global_stats_video: (mean, std) tuple for video, or None
        augment: unused (kept for API compatibility)
    """

    def __init__(
        self,
        samples: List[Tuple],
        sensor_max_len: int = 256,
        video_max_len: int = 128,
        feature_type: str = "raw+velocity",
        landmark_set: str = "hands_legs_hips",
        normalization_type: str = "global",
        global_stats_sensor: Tuple = None,
        global_stats_video: Tuple = None,
        augment: bool = False,
        augment_minority: bool = False,
        minority_classes: set = None,
        augment_minority_params: dict = None,
    ):
        self.samples = samples
        self.sensor_max_len = sensor_max_len
        self.video_max_len = video_max_len
        self.feature_type = feature_type
        self.landmark_set = landmark_set
        self.normalization_type = normalization_type
        self.global_stats_sensor = global_stats_sensor
        self.global_stats_video = global_stats_video
        self.augment = augment
        self.augment_minority = augment_minority
        self.minority_classes = minority_classes or set()
        self.augment_minority_params = augment_minority_params or {}
        
        # Extract augmentation parameters
        self.amp_scale_range = self.augment_minority_params.get("amplitude_scale_range", [0.8, 1.2])
        self.time_warp_sigma = self.augment_minority_params.get("time_warp_sigma", 0.15)
        self.landmark_rotation_range = self.augment_minority_params.get("landmark_rotation_range", 5.0)
        self.landmark_scale_range = self.augment_minority_params.get("landmark_scale_range", [0.9, 1.1])
        
        self.landmark_indices = get_landmark_indices(landmark_set)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        """Return ((x_sensor, x_video), label)."""
        label, paired_data = self.samples[idx]
        sensor_raw, video_raw = paired_data

        # ── Process sensor ───────────────────────────────────────────────────
        sensor_raw = sensor_raw.astype(np.float32)

        if self.feature_type == "hand_crafted":
            x_sensor = extract_imu_features(sensor_raw).reshape(1, -1)
        else:
            x_sensor = add_velocity_sensor(sensor_raw)  # (T, 12)

            if self.normalization_type == "global":
                global_mean, global_std = self.global_stats_sensor
                x_sensor = (x_sensor - global_mean) / global_std
            else:
                mu = x_sensor.mean(axis=0, keepdims=True)
                std = x_sensor.std(axis=0, keepdims=True) + 1e-8
                x_sensor = (x_sensor - mu) / std

            T = x_sensor.shape[0]
            if T > self.sensor_max_len:
                x_sensor = x_sensor[:self.sensor_max_len]
            elif T < self.sensor_max_len:
                pad = np.zeros((self.sensor_max_len - T, x_sensor.shape[1]), dtype=np.float32)
                x_sensor = np.vstack([x_sensor, pad])

        # ── Process video ────────────────────────────────────────────────────
        video_raw = video_raw.astype(np.float32)

        # Filter to selected landmarks FIRST
        col_indices = []
        for i in self.landmark_indices:
            col_indices.extend([i * 2, i * 2 + 1])
        video_filtered = video_raw[:, col_indices]  # (T, len(landmarks)*2)

        # Extract raw wrist deltas AFTER filtering (unnormalized, preserve sign)
        # Must be computed on filtered data for correct landmark indexing
        wrist_deltas = extract_wrist_deltas(video_filtered, self.landmark_indices)  # (T, 2)

        x_video = add_velocity_video(video_filtered)  # (T, len(landmarks)*4)

        if self.normalization_type == "global":
            global_mean, global_std = self.global_stats_video
            x_video = (x_video - global_mean) / global_std
        else:
            mu = x_video.mean(axis=0, keepdims=True)
            std = x_video.std(axis=0, keepdims=True) + 1e-8
            x_video = (x_video - mu) / std

        # Append unnormalized wrist deltas
        x_video = np.concatenate([x_video, wrist_deltas], axis=-1)

        T = x_video.shape[0]
        if T > self.video_max_len:
            x_video = x_video[:self.video_max_len]
        elif T < self.video_max_len:
            pad = np.zeros((self.video_max_len - T, x_video.shape[1]), dtype=np.float32)
            x_video = np.vstack([x_video, pad])

        return (torch.from_numpy(x_sensor.astype(np.float32)),
                torch.from_numpy(x_video.astype(np.float32))), label
