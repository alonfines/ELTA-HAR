"""
Test Transformer Model Evaluation (Load .pt Weights)

Loads per-fold Transformer checkpoint and evaluates on held-out subject.
Uses LOSO cross-validation to report aggregate metrics (mean ± std).

Supports both sensor and video modalities.

Usage:
    python3 test.py --modality sensor
    python3 test.py --modality video
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from data.sensor_dataset import SensorDataset, load_sensor_samples, compute_global_stats
from data.video_dataset import PoseDataset, load_video_samples, compute_global_stats_video, get_landmark_indices
from data.fusion_dataset import FusionDataset, extract_matched_keys
from models.model import TransformerClassifier
from models.fusion import FusionTransformerClassifier
from utils import load_action_names, load_config, save_confusion_matrix

warnings.filterwarnings("ignore")

# ── Jupyter detection ──────────────────────────────────────────────────────────
def is_jupyter():
    """Auto-detect if running in Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except:
        return False

# ── Parse arguments ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--modality", choices=["sensor", "video", "fusion"], default="sensor")
parser.add_argument("--analyse_failure", action="store_true",
                    help="Analyze and visualize failure cases")
args = parser.parse_args()

# ── Load config ────────────────────────────────────────────────────────────────
cfg = load_config(f"configs/{args.modality}.yaml")
SUBSET = cfg.subset

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "Sample_Code"
OUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = ROOT / cfg.checkpoint_dir
OUT_DIR.mkdir(exist_ok=True)

# ── Label map (needed before modality setup) ───────────────────────────────────
label_map = {a: i for i, a in enumerate(SUBSET)}

# ── Modality-specific setup ────────────────────────────────────────────────────
if args.modality == "sensor":
    INERTIAL_DIR = ROOT / cfg.inertial_dir
    in_dim = cfg.in_dim
    dataset_cls = SensorDataset
    cm_cmap = "Blues"
    title_prefix = "SensorTransformer"
    cm_filename = "test_sensor_confusion.png"
    feature_type = getattr(cfg, "feature_type", "raw+velocity")
    normalization_type = getattr(cfg, "normalization_type", "per_sample")
    label_map_sensor = {a: i for i, a in enumerate(cfg.subset)}
    # NOTE: global_stats will be computed per-fold inside loop (no data leakage)
    dataset_kwargs = {"max_len": cfg.max_len, "feature_type": feature_type,
                     "normalization_type": normalization_type}

elif args.modality == "video":
    from video_pretraining import ensure_pose_cache

    KALMAN_CACHE = ROOT / cfg.kalman_cache
    ensure_pose_cache(ROOT, cfg, cfg.subset, label_map)

    in_dim = cfg.in_dim
    dataset_cls = PoseDataset
    cm_cmap = "Greens"
    title_prefix = "PoseTransformer"
    cm_filename = "test_video_confusion.png"
    landmark_set = getattr(cfg, "landmark_set", "all")
    normalization_type = getattr(cfg, "normalization_type", "per_sample")
    label_map_video = {a: i for i, a in enumerate(cfg.subset)}
    # NOTE: global_stats will be computed per-fold inside loop (no data leakage)
    dataset_kwargs = {"max_len": cfg.max_len, "landmark_set": landmark_set,
                     "normalization_type": normalization_type}

elif args.modality == "fusion":
    from video_pretraining import ensure_pose_cache

    INERTIAL_DIR = ROOT / cfg.inertial_dir
    KALMAN_CACHE = ROOT / cfg.kalman_cache
    ensure_pose_cache(ROOT, cfg, cfg.subset, label_map)

    in_dim_sensor = cfg.in_dim_sensor
    in_dim_video = cfg.in_dim_video
    dataset_cls = FusionDataset
    cm_cmap = "RdPu"
    title_prefix = "FusionTransformer"
    cm_filename = "test_fusion_confusion.png"
    feature_type = getattr(cfg, "feature_type", "raw+velocity")
    landmark_set = getattr(cfg, "landmark_set", "hands_legs_hips")
    normalization_type = getattr(cfg, "normalization_type", "per_sample")
    sensor_max_len = getattr(cfg, "max_len_sensor", 256)
    video_max_len = getattr(cfg, "max_len_video", 128)

# Per-fold checkpoint naming
def fold_ckpt_path(subject: int) -> Path:
    return CHECKPOINT_DIR / f"fold_s{subject}.pt"

# ── Plot display wrapper (save + optional inline) ────────────────────────────────
def display_and_save(fig, filepath, title=""):
    """Save plot to file and display inline if in Jupyter."""
    plt.savefig(filepath, dpi=150)
    print(f"✓ {title} saved to {filepath}")
    if is_jupyter():
        plt.show()
    plt.close()

# ── Setup device ───────────────────────────────────────────────────────────────
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Device: {device}")
print(f"Modality: {args.modality.upper()}")

# ── Load action names ──────────────────────────────────────────────────────────
action_names = load_action_names(SAMPLE_DIR)
CLASS_NAMES = [action_names[a] for a in SUBSET]

# ── Load dataset ───────────────────────────────────────────────────────────────
print(f"\nLoading {args.modality} dataset...")
if args.modality == "sensor":
    samples = load_sensor_samples(INERTIAL_DIR, SUBSET, label_map)
elif args.modality == "video":
    samples = load_video_samples(KALMAN_CACHE, SUBSET, label_map)
elif args.modality == "fusion":
    sensor_samples = load_sensor_samples(INERTIAL_DIR, SUBSET, label_map)
    video_samples = load_video_samples(KALMAN_CACHE, SUBSET, label_map)
    matched_samples = extract_matched_keys(sensor_samples, video_samples)
    samples = matched_samples

subjects = sorted(set(s for s, _, _ in samples))
print(f"  {len(samples)} samples | {len(subjects)} subjects | {len(SUBSET)} classes")

if len(samples) == 0:
    print(f"\n❌ No samples found for {args.modality} modality")
    if args.modality == "video":
        print(f"   Run video_pretraining.py first to generate kalman pose cache")
    sys.exit(1)

# ── Verify all per-fold checkpoints exist before starting ──────────────────────
missing = [s for s in subjects if not fold_ckpt_path(s).exists()]
if missing:
    print(f"\n❌ Missing per-fold checkpoints for subject(s): {missing}")
    print(f"   Expected files in: {CHECKPOINT_DIR}")
    print(f"   Pattern: fold_s{{subject}}.pt   (one per held-out subject)")
    print("   Run training first:  python3 train.py --modality sensor --config configs/sensor.yaml")
    sys.exit(1)

# ── Build a model shell once; reload weights per fold ──────────────────────────
def build_model():
    if args.modality == "fusion":
        return FusionTransformerClassifier(
            n_classes=len(SUBSET),
            in_dim_sensor=in_dim_sensor,
            in_dim_video=in_dim_video,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers,
            dropout=cfg.model.dropout,
            d_fusion=cfg.model.d_fusion,
        )
    else:
        return TransformerClassifier(
            in_dim=in_dim,
            n_classes=len(SUBSET),
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers,
            dropout=cfg.model.dropout,
        )

model = build_model().to(device)

# ── LOSO inference (one checkpoint per fold) ───────────────────────────────────
y_true_all: list[int] = []
y_pred_all: list[int] = []
fold_accuracies: list[float] = []
fold_f1_scores: list[float] = []

with torch.no_grad():
    for test_subject in subjects:
        ckpt_file = fold_ckpt_path(test_subject)
        state = torch.load(ckpt_file, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Compute global stats on TRAINING data only (no data leakage from test subject)
        fold_dataset_kwargs = dict(dataset_kwargs) if args.modality != "fusion" else {}

        if normalization_type == "global":
            train_subject_samples = [(s, y, data) for s, y, data in samples if s != test_subject]
            if args.modality == "sensor":
                fold_dataset_kwargs["global_stats"] = compute_global_stats(train_subject_samples)
            elif args.modality == "video":
                landmark_indices = get_landmark_indices(landmark_set)
                fold_dataset_kwargs["global_stats"] = compute_global_stats_video(train_subject_samples, landmark_indices)
            elif args.modality == "fusion":
                train_sensor_samples = [(s, y, data[0]) for s, y, data in train_subject_samples
                                       if isinstance(data, tuple) and len(data) == 2]
                train_video_samples = [(s, y, data[1]) for s, y, data in train_subject_samples
                                      if isinstance(data, tuple) and len(data) == 2]
                fold_dataset_kwargs["global_stats_sensor"] = compute_global_stats(train_sensor_samples)
                landmark_indices = get_landmark_indices(landmark_set)
                fold_dataset_kwargs["global_stats_video"] = compute_global_stats_video(train_video_samples, landmark_indices)
                fold_dataset_kwargs["sensor_max_len"] = sensor_max_len
                fold_dataset_kwargs["video_max_len"] = video_max_len
                fold_dataset_kwargs["feature_type"] = feature_type
                fold_dataset_kwargs["landmark_set"] = landmark_set
                fold_dataset_kwargs["normalization_type"] = normalization_type

        test_samples = [(label, raw) for s, label, raw in samples if s == test_subject]
        test_data = dataset_cls(test_samples, **fold_dataset_kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

        y_fold_true: list[int] = []
        y_fold_pred: list[int] = []

        for batch in test_loader:
            if args.modality == "fusion":
                (x_sensor, x_video), y = batch
                x_sensor = x_sensor.to(device)
                x_video = x_video.to(device)
                logits = model(x_sensor, x_video)
            else:
                x, y = batch
                x = x.to(device)
                logits = model(x)

            preds = logits.argmax(dim=1).cpu().numpy()
            y_np = y.numpy()
            y_fold_true.extend(y_np)
            y_fold_pred.extend(preds)

        # Per-fold metrics
        fold_acc = accuracy_score(y_fold_true, y_fold_pred)
        fold_f1 = f1_score(y_fold_true, y_fold_pred, average="macro")
        fold_accuracies.append(fold_acc)
        fold_f1_scores.append(fold_f1)

        # Pool for confusion matrix
        y_true_all.extend(y_fold_true)
        y_pred_all.extend(y_fold_pred)

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# ── Aggregate (mean ± std over folds) ──────────────────────────────────────────
acc_mean, acc_std = float(np.mean(fold_accuracies)), float(np.std(fold_accuracies))
f1_mean, f1_std = float(np.mean(fold_f1_scores)), float(np.std(fold_f1_scores))

# ── Confusion matrix (pooled predictions across folds) ─────────────────────────
path = OUT_DIR / cm_filename
cm = save_confusion_matrix(
    y_true_all, y_pred_all, CLASS_NAMES,
    title=f"{title_prefix} — Accuracy={acc_mean:.3f}±{acc_std:.3f} | F1={f1_mean:.3f}±{f1_std:.3f}",
    path=path,
    cmap=cm_cmap,
)
print(f"✓ Confusion matrix saved to {path}")
if is_jupyter():
    plt.show()
plt.close()

# ── Helper: parse failure_case from config ────────────────────────────────────────
def parse_failure_case(cfg, label_map):
    """Parse failure_case config section to get action indices.

    Returns:
        tuple: (true_idx, pred_idx) if config has action_ids, else (None, None)
    """
    if hasattr(cfg, 'failure_case') and hasattr(cfg.failure_case, 'action_ids'):
        action_ids = cfg.failure_case.action_ids
        true_idx = label_map[action_ids[0]]
        pred_idx = label_map[action_ids[1]]
        return true_idx, pred_idx
    return None, None


# ── Failure case analysis (modal-specific) ─────────────────────────────────────────
if args.analyse_failure:
    true_idx, pred_idx = parse_failure_case(cfg, label_map)
    if true_idx is None:
        # Fall back to most confused off-diagonal pair from confusion matrix
        cm_off_diag = cm.copy()
        np.fill_diagonal(cm_off_diag, 0)
        true_idx, pred_idx = np.unravel_index(np.argmax(cm_off_diag), cm_off_diag.shape)

    true_action = SUBSET[true_idx]
    pred_action = SUBSET[pred_idx]
    action_pair = {true_idx: action_names[true_action], pred_idx: action_names[pred_action]}

    misclassified = []
    with torch.no_grad():
        for test_subject in subjects:
            ckpt_file = fold_ckpt_path(test_subject)
            state = torch.load(ckpt_file, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()

            # Compute fold-specific global stats (no data leakage from test subject)
            fold_dataset_kwargs = dict(dataset_kwargs) if args.modality != "fusion" else {}
            if normalization_type == "global":
                train_subject_samples = [(s, y, data) for s, y, data in samples if s != test_subject]
                if args.modality == "sensor":
                    fold_dataset_kwargs["global_stats"] = compute_global_stats(train_subject_samples)
                elif args.modality == "video":
                    landmark_indices = get_landmark_indices(landmark_set)
                    fold_dataset_kwargs["global_stats"] = compute_global_stats_video(train_subject_samples, landmark_indices)
                elif args.modality == "fusion":
                    train_sensor_samples = [(s, y, data[0]) for s, y, data in train_subject_samples
                                           if isinstance(data, tuple) and len(data) == 2]
                    train_video_samples = [(s, y, data[1]) for s, y, data in train_subject_samples
                                          if isinstance(data, tuple) and len(data) == 2]
                    fold_dataset_kwargs["global_stats_sensor"] = compute_global_stats(train_sensor_samples)
                    landmark_indices = get_landmark_indices(landmark_set)
                    fold_dataset_kwargs["global_stats_video"] = compute_global_stats_video(train_video_samples, landmark_indices)
                    fold_dataset_kwargs["sensor_max_len"] = sensor_max_len
                    fold_dataset_kwargs["video_max_len"] = video_max_len
                    fold_dataset_kwargs["feature_type"] = feature_type
                    fold_dataset_kwargs["landmark_set"] = landmark_set
                    fold_dataset_kwargs["normalization_type"] = normalization_type

            test_samples = [(label, raw) for s, label, raw in samples if s == test_subject]
            test_samples_pair = [(i, y, raw) for i, (y, raw) in enumerate(test_samples) if y in [true_idx, pred_idx]]

            if not test_samples_pair:
                continue

            test_data_samples = [(y, raw) for _, y, raw in test_samples_pair]
            test_data = dataset_cls(test_data_samples, **fold_dataset_kwargs)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
            sample_idx_in_pair = 0

            for batch in test_loader:
                if args.modality == "fusion":
                    (x_sensor, x_video), y = batch
                    x_sensor = x_sensor.to(device)
                    x_video = x_video.to(device)
                    logits = model(x_sensor, x_video)
                else:
                    x, y = batch
                    x = x.to(device)
                    logits = model(x)

                preds = logits.argmax(dim=1).cpu().numpy()
                y_batch = y.numpy()

                for i in range(len(y_batch)):
                    y_true = y_batch[i]
                    y_pred = preds[i]
                    if y_pred != y_true:
                        _, _, raw = test_samples_pair[sample_idx_in_pair]
                        misclassified.append((test_subject, int(y_true), int(y_pred), raw))
                    sample_idx_in_pair += 1

    if misclassified:
        # Filter to ensure we have the correct action pair
        s, y_true, y_pred, raw = misclassified[0]
        if y_true not in action_pair or y_pred not in action_pair:
            if len(misclassified) > 1:
                misclassified = misclassified[1:]
                if misclassified:
                    s, y_true, y_pred, raw = misclassified[0]

        if misclassified:
            s, y_true, y_pred, raw = misclassified[0]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"Failure Case — Subject {s}: true={action_pair[y_true]}, predicted={action_pair[y_pred]}",
                fontweight="bold"
            )

            if args.modality == "sensor":
                # Sensor: Plot 6-channel IMU data
                t = np.arange(len(raw))

                for i, lbl in enumerate(["Ax", "Ay", "Az"]):
                    axes[0].plot(t, raw[:, i], label=lbl)
                axes[0].set_title("Accelerometer")
                axes[0].set_xlabel("Sample")
                axes[0].set_ylabel("g")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                for i, lbl in enumerate(["Gx", "Gy", "Gz"]):
                    axes[1].plot(t, raw[:, 3 + i], label=lbl)
                axes[1].set_title("Gyroscope")
                axes[1].set_xlabel("Sample")
                axes[1].set_ylabel("°/s")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                path = OUT_DIR / "test_sensor_failure.png"
                plt.savefig(path, dpi=150)
                print(f"✓ Failure case plot saved to {path}")

            elif args.modality == "video":
                # Video: Plot wrist trajectories from raw (T, 66) pose data
                t = np.arange(len(raw))

                # Left Wrist (landmark 15): columns 30 (X), 31 (Y)
                left_wrist_x = raw[:, 30]
                left_wrist_y = raw[:, 31]

                # Right Wrist (landmark 16): columns 32 (X), 33 (Y)
                right_wrist_x = raw[:, 32]
                right_wrist_y = raw[:, 33]

                axes[0].plot(t, left_wrist_x, label="Left Wrist X", linewidth=1.5)
                axes[0].plot(t, left_wrist_y, label="Left Wrist Y", linewidth=1.5)
                axes[0].set_title("Left Wrist Trajectory")
                axes[0].set_xlabel("Frame")
                axes[0].set_ylabel("Position (normalized)")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(t, right_wrist_x, label="Right Wrist X", linewidth=1.5)
                axes[1].plot(t, right_wrist_y, label="Right Wrist Y", linewidth=1.5)
                axes[1].set_title("Right Wrist Trajectory")
                axes[1].set_xlabel("Frame")
                axes[1].set_ylabel("Position (normalized)")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                path = OUT_DIR / "test_video_failure.png"
                plt.savefig(path, dpi=150)
                print(f"✓ Failure case plot saved to {path}")

            elif args.modality == "fusion":
                print(f"  Failure case: true={action_pair[y_true]}, predicted={action_pair[y_pred]} (detailed visualization skipped for fusion)")

            if is_jupyter():
                plt.show()
            plt.close()
    else:
        print("  No misclassifications found — classes fully separated.")
else:
    print("  Skipping failure case analysis (use --analyse_failure to enable)")
