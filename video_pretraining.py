import os
import re
import urllib.request
import warnings
from pathlib import Path
from typing import List, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

warnings.filterwarnings("ignore")

def _get_pose_model_path() -> str:
    """Download pose_landmarker.task model if not cached locally."""
    cache_dir = os.path.expanduser("~/.cache/mediapipe")
    model_path = os.path.join(cache_dir, "pose_landmarker.task")

    if not os.path.exists(model_path):
        os.makedirs(cache_dir, exist_ok=True)
        model_url = "https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task"
        print(f"  Downloading pose_landmarker.task model (~5.5 MB)...")
        urllib.request.urlretrieve(model_url, model_path)

    return model_path

def extract_poses_from_video(video_path: Path, landmarker: PoseLandmarker) -> np.ndarray:
    """Extract 33 pose landmarks from a single video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ⚠ Failed to open {video_path.name}")
        return np.zeros((0, 33, 2), dtype=np.float32)

    poses = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            detection_result = landmarker.detect(mp_image)

            if detection_result.pose_landmarks:
                landmarks = np.array([
                    [lm.x, lm.y] for lm in detection_result.pose_landmarks[0]
                ], dtype=np.float32)
                poses.append(landmarks)
            else:
                # Use zeros if detection failed (handled later by smoothing/interpolation)
                poses.append(np.zeros((33, 2), dtype=np.float32))
        except Exception:
            poses.append(np.zeros((33, 2), dtype=np.float32))

    cap.release()
    return np.array(poses, dtype=np.float32) if poses else np.zeros((0, 33, 2), dtype=np.float32)

def apply_kalman_smoothing(poses: np.ndarray, process_variance: float = 1e-5) -> np.ndarray:
    """Apply Kalman filtering to smooth pose landmarks (T, 33, 2)."""
    if len(poses) == 0:
        return poses

    T, num_landmarks, num_coords = poses.shape
    smoothed = np.zeros_like(poses)

    for lm_idx in range(num_landmarks):
        for coord_idx in range(num_coords):
            signal = poses[:, lm_idx, coord_idx]
            
            # Initialization
            x_est = signal[0]
            p_est = 1.0
            measurement_variance = 1e-3

            for t in range(T):
                # Prediction
                x_pred = x_est
                p_pred = p_est + process_variance

                # Update
                K = p_pred / (p_pred + measurement_variance)
                x_est = x_pred + K * (signal[t] - x_pred)
                p_est = (1 - K) * p_pred

                smoothed[t, lm_idx, coord_idx] = x_est
    return smoothed

def normalize_poses(poses: np.ndarray) -> np.ndarray:
    """Normalize poses to hip center and shoulder width scale."""
    if len(poses) == 0:
        return poses

    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

    normalized = poses.copy()
    for t in range(len(poses)):
        # Calculate center as midpoint of hips
        center = (poses[t, LEFT_HIP] + poses[t, RIGHT_HIP]) / 2
        # Calculate scale as distance between shoulders
        dist = np.linalg.norm(poses[t, RIGHT_SHOULDER] - poses[t, LEFT_SHOULDER])
        scale = dist if dist > 1e-6 else 1.0
        
        normalized[t] = (poses[t] - center) / scale
    return normalized

def generate_pose_cache(root_dir: Path, rgb_dirs: List[str], output_cache_dir: Path, subset: List[int]):
    """Main pipeline: Extract -> Kalman -> Normalize -> Cache."""
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    subset_set = set(subset)
    fname_re = re.compile(r"a(\d+)_s(\d+)_t(\d+)_color\.avi")

    video_files = []
    for rgb_dir in rgb_dirs:
        path = root_dir / rgb_dir
        if path.exists():
            video_files.extend(sorted(path.glob("*_color.avi")))

    if not video_files:
        print("❌ No video files found.")
        return

    model_path = _get_pose_model_path()
    options = PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=model_path),
        running_mode=PoseLandmarkerOptions.running_mode.IMAGE
    )

    print(f"🚀 Starting preprocessing for {len(video_files)} videos...")
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        for i, video_path in enumerate(video_files, 1):
            m = fname_re.match(video_path.name)
            if not m or int(m.group(1)) not in subset_set:
                continue

            cache_path = output_cache_dir / f"a{m.group(1)}_s{m.group(2)}_t{m.group(3)}_poses.npy"
            if cache_path.exists():
                continue

            # 1. Extraction
            raw_poses = extract_poses_from_video(video_path, landmarker)
            if len(raw_poses) == 0:
                continue

            # 2. Kalman Smoothing (removes jitter)
            smoothed_poses = apply_kalman_smoothing(raw_poses, process_variance=1e-5)

            # 3. Normalization (View Invariance)
            norm_poses = normalize_poses(smoothed_poses)

            # 4. Save as (T, 66)
            T = norm_poses.shape[0]
            np.save(str(cache_path), norm_poses.reshape(T, -1))
            
            if i % 10 == 0:
                print(f"  Processed {i}/{len(video_files)}...")

    print(f"✅ Preprocessing complete. Cached files in: {output_cache_dir}")

def ensure_pose_cache(root_dir: Path, cfg, subset: List[int], label_map: dict):
    """Entry point used by train.py."""
    kalman_cache = root_dir / cfg.kalman_cache
    if not any(kalman_cache.glob("*.npy")):
        RGB_DIRS = ["RGB-part1", "RGB-part2", "RGB-part3", "RGB-part4"]
        generate_pose_cache(root_dir, RGB_DIRS, kalman_cache, subset)