"""
IMU sequence dataset — loads raw inertial .mat files,
pads/truncates to a fixed length, and applies z-score normalization.
"""

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io
import torch
from scipy.fft import rfft, rfftfreq
from torch.utils.data import Dataset

_FS      = 50.0
_FFT_LEN = 256


def extract_imu_features(iner: np.ndarray) -> np.ndarray:
    """36-dim feature vector: per channel (6) × [mean, std, RMS, energy, dom_freq, spec_entropy]."""
    feats = []
    for ch in range(6):
        x      = iner[:, ch]
        padded = np.zeros(_FFT_LEN)
        padded[:min(len(x), _FFT_LEN)] = x[:_FFT_LEN]
        mag          = np.abs(rfft(padded))
        fqs          = rfftfreq(_FFT_LEN, d=1.0 / _FS)
        dom_freq     = fqs[np.argmax(mag[1:]) + 1]
        psd          = mag ** 2
        psd_norm     = psd / (psd.sum() + 1e-8)
        spec_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
        feats += [x.mean(), x.std(), np.sqrt(np.mean(x**2)), np.sum(x**2),
                  dom_freq, spec_entropy]
    return np.array(feats, dtype=np.float32)


def add_velocity(seq: np.ndarray) -> np.ndarray:
    """Append frame-to-frame differences to position. (T, D) -> (T, 2D)."""
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, vel], axis=-1).astype(np.float32)


def pad_or_truncate(x: np.ndarray, max_len: int) -> np.ndarray:
    T = x.shape[0]
    if T >= max_len:
        return x[:max_len]
    pad = np.zeros((max_len - T, x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


def load_sensor_samples(
    inertial_dir: Path,
    subset: List[int],
    label_map: dict,
) -> List[Tuple[int, int, np.ndarray]]:
    """Return list of (subject, label, raw_iner) from .mat files.

    raw_iner shape: (T, 6)  — 3-axis accelerometer + 3-axis gyroscope
    """
    fname_re   = re.compile(r"a(\d+)_s(\d+)_t(\d+)_")
    subset_set = set(subset)
    samples: List[Tuple[int, int, np.ndarray]] = []

    for path in sorted(inertial_dir.glob("*_inertial.mat")):
        m = fname_re.match(path.name)
        if not m:
            continue
        a, s, _ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a not in subset_set:
            continue
        raw = scipy.io.loadmat(str(path))["d_iner"]
        samples.append((s, label_map[a], raw))

    return samples


def compute_global_stats(samples: List[Tuple[int, int, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global mean and std across all samples for normalization.

    Computes statistics on raw+velocity data (12-dim) for use with raw+velocity mode.
    Returns: (global_mean, global_std) each shape (1, 12)
    """
    all_seqs = [add_velocity(raw.astype(np.float32)) for _, _, raw in samples]
    all_data = np.vstack(all_seqs)
    global_mean = all_data.mean(axis=0, keepdims=True)
    global_std = all_data.std(axis=0, keepdims=True) + 1e-8
    return global_mean, global_std


class SensorDataset(Dataset):
    """IMU sequence dataset supporting multiple feature extraction and normalization modes.

    Args:
        samples:            list of (label, raw_iner) where raw_iner is (T, 6)
        max_len:            pad/truncate sequences to this many timesteps
        augment:            unused (kept for API compatibility)
        feature_type:       "raw+velocity" (12-dim) or "hand_crafted" (36-dim statistical features)
        normalization_type: "per_sample" (erases amplitude) or "global" (preserves amplitude differences)
        global_stats:       tuple (mean, std) for global normalization; if None, computed from samples

    Returns per item (raw+velocity, per_sample):
        x:     float32 tensor (max_len, 12)  — z-score normalised position (6) + velocity (6)

    Returns per item (raw+velocity, global):
        x:     float32 tensor (max_len, 12)  — globally scaled position (6) + velocity (6)

    Returns per item (hand_crafted):
        x:     float32 tensor (1, 36)  — statistical features per channel
        label: int class index
    """

    def __init__(
        self,
        samples: List[Tuple[int, np.ndarray]],
        max_len: int = 256,
        augment: bool = False,
        feature_type: str = "raw+velocity",
        normalization_type: str = "per_sample",
        global_stats: tuple = None,
    ):
        self.samples = samples
        self.max_len = max_len
        self.augment = augment
        self.feature_type = feature_type
        self.normalization_type = normalization_type

        assert feature_type in ["raw+velocity", "hand_crafted"], f"Unknown feature_type: {feature_type}"
        assert normalization_type in ["per_sample", "global"], f"Unknown normalization_type: {normalization_type}"

        # For global normalization, compute or use provided stats
        if normalization_type == "global":
            if global_stats is None:
                self.global_mean, self.global_std = compute_global_stats(samples)
            else:
                self.global_mean, self.global_std = global_stats
        else:
            self.global_mean = None
            self.global_std = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label, raw = self.samples[idx]
        raw = raw.astype(np.float32)

        if self.feature_type == "hand_crafted":
            # Extract 36-dim hand-crafted statistical features (not temporal)
            x = extract_imu_features(raw)  # (36,)
            # Return as (1, 36) to be compatible with transformer input expectations
            x = x.reshape(1, -1)
        else:
            # raw+velocity: temporal sequence
            x = raw
            # Add velocity features (position + velocity)
            x = add_velocity(x)

            # Truncate if longer than max_len (before normalization)
            T = x.shape[0]
            if T > self.max_len:
                x = x[:self.max_len]
                T = self.max_len

            # Normalize (per-sample or global)
            if self.normalization_type == "global":
                # Global normalization: preserves amplitude differences between actions
                x = (x - self.global_mean) / self.global_std
            else:
                # Per-sample normalization: erases amplitude information
                mu  = x.mean(axis=0, keepdims=True)
                std = x.std(axis=0, keepdims=True) + 1e-8
                x = (x - mu) / std

            # Pad after normalization (padding zeros are now non-zero after normalization)
            if T < self.max_len:
                padding = np.zeros((self.max_len - T, x.shape[1]), dtype=np.float32)
                x = np.vstack([x, padding])

        return torch.from_numpy(x), label