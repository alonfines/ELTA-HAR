"""
Pose sequence dataset — loads Kalman-smoothed cached sequences,
drops low-importance features, and appends frame-to-frame velocity.
"""

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def add_velocity(seq: np.ndarray) -> np.ndarray:
    """Append frame-to-frame differences to position. (T, D) -> (T, 2D)."""
    vel = np.diff(seq, axis=0, prepend=seq[:1, :])
    return np.concatenate([seq, vel], axis=-1).astype(np.float32)


def extract_wrist_deltas(seq: np.ndarray, landmark_indices: List[int]) -> np.ndarray:
    """Extract raw frame-to-frame x-deltas for both wrists (unnormalized, preserve sign).

    Args:
        seq: shape (T, D) where D = len(landmark_indices) * 2 (already filtered to selected landmarks)
        landmark_indices: list of landmark indices used (from get_landmark_indices)

    Returns:
        shape (T, 2) containing [left_wrist_Δx, right_wrist_Δx] for each frame, unnormalized
    """
    # Wrist landmarks in full 33-landmark set are [15, 16]
    # (NOT 9, 10 which are mouth corners)
    wrist_indices_full = [15, 16]

    # Find positions of wrists in the filtered landmark set
    wrist_positions = []
    for wrist_idx in wrist_indices_full:
        if wrist_idx in landmark_indices:
            pos = landmark_indices.index(wrist_idx)
            wrist_positions.append(pos)

    if len(wrist_positions) < 2:
        # If wrists not in landmark set, return zeros
        return np.zeros((seq.shape[0], 2), dtype=np.float32)

    # Extract x-coordinates (every other coordinate, starting at even indices)
    # Each landmark has 2 coords (x, y), so x is at position*2
    wrist_x_coords = []
    for pos in wrist_positions:
        x_col = pos * 2
        wrist_x_coords.append(seq[:, x_col])

    # Compute frame-to-frame x-deltas (preserve sign)
    deltas = []
    for x_coords in wrist_x_coords:
        delta = np.diff(x_coords, prepend=x_coords[0])  # First frame delta is 0
        deltas.append(delta)

    return np.column_stack(deltas).astype(np.float32)


def get_landmark_indices(landmark_set: str) -> List[int]:
    """Get MediaPipe Pose landmark indices by body part.

    Full 33 landmarks:
      0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 7-8: elbows, 9-10: wrists,
      11-12: hips, 13-14: knees, 15-16: ankles, 17-18: toes, 19-32: hand landmarks
    """
    if landmark_set == "all":
        return list(range(33))
    elif landmark_set == "hands_legs_hips":
        # Wrists, hand landmarks, hips, knees, ankles, toes
        return [9, 10] + list(range(11, 33))  # [9,10,11,12,...,32] = 24 landmarks
    elif landmark_set == "upper_body":
        # Shoulders, elbows, wrists, hand landmarks
        return [5, 6, 7, 8, 9, 10] + list(range(19, 33))  # 22 landmarks
    elif landmark_set == "lower_body":
        # Hips, knees, ankles, toes
        return list(range(11, 19))  # [11-18] = 8 landmarks
    else:
        raise ValueError(f"Unknown landmark_set: {landmark_set}")


def load_video_samples(
    kalman_cache: Path,
    subset: List[int],
    label_map: dict,
) -> List[Tuple[int, int, np.ndarray]]:
    """Return list of (subject, label, seq) from the cache.

    seq shape: (T, 66) — all 33 landmarks × 2 coordinates (no feature dropping)
    """
    fname_re   = re.compile(r"a(\d+)_s(\d+)_t(\d+)_")
    subset_set = set(subset)
    samples: List[Tuple[int, int, np.ndarray]] = []

    for path in sorted(kalman_cache.glob("*.npy")):
        m = fname_re.match(path.stem)
        if not m:
            continue
        a, s, _ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a not in subset_set:
            continue
        seq = np.load(str(path))
        samples.append((s, label_map[a], seq))

    return samples


def compute_global_stats_video(
    samples: List[Tuple[int, int, np.ndarray]],
    landmark_indices: List[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global mean and std across all pose samples for normalization.

    Computes statistics on position+velocity data for use with pose data.
    Returns: (global_mean, global_std) each shape (1, D*2) where D = len(landmark_indices)
    """
    all_seqs = []
    for _, _, seq in samples:
        s = seq.astype(np.float32)
        if landmark_indices is not None:
            # Convert landmark indices to coordinate column indices (each landmark has 2 coords)
            coord_indices = []
            for lm_idx in landmark_indices:
                coord_indices.extend([lm_idx * 2, lm_idx * 2 + 1])
            s = s[:, coord_indices]
        s_with_vel = add_velocity(s)
        all_seqs.append(s_with_vel)
    all_data = np.vstack(all_seqs)
    global_mean = all_data.mean(axis=0, keepdims=True)
    global_std = all_data.std(axis=0, keepdims=True) + 1e-8
    return global_mean, global_std


class PoseDataset(Dataset):
    """Pose sequence dataset with landmark filtering, normalization, and directional features.

    Args:
        samples:            list of (label, seq) where seq is (T, 66)
        augment:            unused (kept for API compatibility)
        max_len:            pad/truncate sequences to this length (use median if None)
        landmark_set:       "all" (33 landmarks), "hands_legs_hips" (24), "upper_body" (22), "lower_body" (8)
        normalization_type: "per_sample" (erases amplitude) or "global" (preserves amplitude differences)
        global_stats:       tuple (mean, std) for global normalization; if None, computed from samples

    Returns per item:
        x:     float32 tensor (max_len, D*2+2)  — [position (D) + velocity (D) + raw_wrist_deltas (2)]
               where D = 66 (all)→132+2, 48 (hands_legs_hips)→96+2, 44 (upper_body)→88+2, 16 (lower_body)→32+2
               Position and velocity are normalized; wrist deltas are unnormalized (preserve direction)
        label: int class index
    """

    def __init__(
        self,
        samples: List[Tuple[int, np.ndarray]],
        augment: bool = False,
        max_len: int = None,
        landmark_set: str = "all",
        normalization_type: str = "per_sample",
        global_stats: tuple = None,
    ):
        self.samples = samples
        self.augment = augment
        self.normalization_type = normalization_type
        self.landmark_set = landmark_set
        self.landmark_indices = get_landmark_indices(landmark_set)

        assert normalization_type in ["per_sample", "global"], f"Unknown normalization_type: {normalization_type}"

        if max_len is None:
            # Use median sequence length if not specified
            lengths = [seq.shape[0] for _, seq in samples]
            self.max_len = int(np.median(lengths))
        else:
            self.max_len = max_len

        # For global normalization, compute or use provided stats
        if normalization_type == "global":
            if global_stats is None:
                self.global_mean, self.global_std = compute_global_stats_video(samples, self.landmark_indices)
            else:
                self.global_mean, self.global_std = global_stats
        else:
            self.global_mean = None
            self.global_std = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label, seq = self.samples[idx]
        x = seq.astype(np.float32)

        # Select landmarks (if not using all)
        if self.landmark_set != "all":
            # Convert landmark indices to coordinate column indices
            coord_indices = []
            for lm_idx in self.landmark_indices:
                coord_indices.extend([lm_idx * 2, lm_idx * 2 + 1])
            x = x[:, coord_indices]

        # Extract raw wrist x-deltas BEFORE velocity/normalization (preserve sign)
        wrist_deltas = extract_wrist_deltas(x, self.landmark_indices)

        # Add velocity (on real frames only)
        x = add_velocity(x)

        # Truncate if longer than max_len (before normalization)
        T = x.shape[0]
        if T > self.max_len:
            x = x[:self.max_len]
            wrist_deltas = wrist_deltas[:self.max_len]
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

        # Pad normalized position+velocity data
        if T < self.max_len:
            padding = np.zeros((self.max_len - T, x.shape[1]), dtype=np.float32)
            x = np.vstack([x, padding])

        # Append raw wrist deltas (unnormalized, preserve sign for directionality)
        # Wrist deltas are already shape (T, 2), pad if needed
        if T < self.max_len:
            wrist_padding = np.zeros((self.max_len - T, 2), dtype=np.float32)
            wrist_deltas = np.vstack([wrist_deltas, wrist_padding])

        x = np.hstack([x, wrist_deltas])

        return torch.from_numpy(x), label
