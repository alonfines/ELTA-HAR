"""
Test Transformer Model Evaluation (Load .pt Weights)

Loads per-fold Transformer checkpoint and evaluates on held-out subject.
Uses LOSO cross-validation to report aggregate metrics (mean ± std).

Supports sensor, video, and fusion modalities with missing-modality testing via dynamic routing.

Usage:
    python3 test.py --modality sensor
    python3 test.py --modality video
    python3 test.py --modality fusion
    python3 test.py --modality fusion --missing-sensor
    python3 test.py --modality fusion --missing-video
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
parser = argparse.ArgumentParser(description="""
Test Transformer Model Evaluation (Load .pt Weights).

Supports three modalities:
  - sensor: Inertial sensor modality only
  - video: Video pose modality only
  - fusion: Bimodal fusion with dynamic routing for missing modalities
    * --missing-sensor: Load video checkpoint only, skip sensor encoder
    * --missing-video: Load sensor checkpoint only, skip video encoder
""")
parser.add_argument("--modality", choices=["sensor", "video", "fusion"], default="sensor")
parser.add_argument("--missing-sensor", action="store_true",
                    help="Test with missing sensor modality (fusion: load video checkpoint only)")
parser.add_argument("--missing-video", action="store_true",
                    help="Test with missing video modality (fusion: load sensor checkpoint only)")
parser.add_argument("--analyse_failure", action="store_true",
                    help="Analyze and visualize failure cases")
parser.add_argument("--conformal", action="store_true",
                    help="Evaluate using split-conformal prediction (requires conformal-trained checkpoints)")
parser.add_argument("--imbalance", action="store_true",
                    help="Load imbalance-trained checkpoints")
parser.add_argument("--weighted_loss", action="store_true",
                    help="Load weighted_loss-trained checkpoints")
parser.add_argument("--augment_minority", action="store_true",
                    help="Load augment_minority-trained checkpoints")
args = parser.parse_args()

# ── Load config ────────────────────────────────────────────────────────────────
cfg = load_config(f"configs/{args.modality}.yaml")
SUBSET = cfg.subset

# Load imbalance config
imbalance_target_actions = getattr(cfg, "imbalance_target_actions", [13, 22])
imbalance_ratio = getattr(cfg, "imbalance_ratio", 0.5)

if args.imbalance:
    print(f"⚠️  Testing IMBALANCE-trained model (ratio={imbalance_ratio})")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "Sample_Code"
OUT_DIR = ROOT / "outputs"
if args.conformal:
    CHECKPOINT_DIR = ROOT / f"checkpoints/conformal/{args.modality}"
elif args.augment_minority and args.imbalance:
    CHECKPOINT_DIR = ROOT / f"checkpoints/imbalance_aug/{args.modality}"
elif args.weighted_loss and args.imbalance:
    CHECKPOINT_DIR = ROOT / f"checkpoints/imbalance_weighted/{args.modality}"
elif args.imbalance:
    CHECKPOINT_DIR = ROOT / f"checkpoints/imbalance/{args.modality}"
else:
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
    suffix = ""
    if args.conformal:
        suffix = "_conformal"
    elif args.augment_minority and args.imbalance:
        suffix = "_imbalance_aug"
    elif args.weighted_loss and args.imbalance:
        suffix = "_imbalance_wce"
    elif args.imbalance:
        suffix = "_imbalance"
    cm_filename = f"test_sensor{suffix}_confusion.png"
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
    suffix = ""
    if args.conformal:
        suffix = "_conformal"
    elif args.augment_minority and args.imbalance:
        suffix = "_imbalance_aug"
    elif args.weighted_loss and args.imbalance:
        suffix = "_imbalance_wce"
    elif args.imbalance:
        suffix = "_imbalance"
    cm_filename = f"test_video{suffix}_confusion.png"
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

    # Colormap based on missing modality scenario
    # Compute WCE suffix
    if args.conformal:
        wce_suffix = "_conformal"
    else:
        wce_suffix = "_imbalance_aug" if (args.augment_minority and args.imbalance) else ("_imbalance_wce" if args.weighted_loss else ("_imbalance" if args.imbalance else ""))

    if args.missing_sensor:
        cm_cmap = "Oranges"
        title_prefix = "FusionTransformer (Missing Sensor)"
        suffix = "_missing_sensor" + wce_suffix
        cm_filename = f"test_fusion{suffix}_confusion.png"
    elif args.missing_video:
        cm_cmap = "Purples"
        title_prefix = "FusionTransformer (Missing Video)"
        suffix = "_missing_video" + wce_suffix
        cm_filename = f"test_fusion{suffix}_confusion.png"
    else:
        cm_cmap = "RdPu"
        title_prefix = "FusionTransformer"
        suffix = "_conformal" if args.conformal else ("_imbalance" if args.imbalance else "")
        cm_filename = f"test_fusion{suffix}_confusion.png"

    feature_type = getattr(cfg, "feature_type", "raw+velocity")
    landmark_set = getattr(cfg, "landmark_set", "hands_legs_hips")
    normalization_type = getattr(cfg, "normalization_type", "per_sample")
    sensor_max_len = getattr(cfg, "max_len_sensor", 256)
    video_max_len = getattr(cfg, "max_len_video", 128)

    dataset_kwargs = {
        "sensor_max_len": sensor_max_len,
        "video_max_len": video_max_len,
        "feature_type": feature_type,
        "landmark_set": landmark_set,
        "normalization_type": normalization_type,
    }

# Per-fold checkpoint naming
def fold_ckpt_path(subject: int, modality: str = None) -> Path:
    if modality is None:
        modality = args.modality
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
all_video_samples = None  # Will be set for fusion modality to compute stats consistently
if args.modality == "sensor":
    samples = load_sensor_samples(INERTIAL_DIR, SUBSET, label_map)
elif args.modality == "video":
    samples = load_video_samples(KALMAN_CACHE, SUBSET, label_map)
elif args.modality == "fusion":
    sensor_samples = load_sensor_samples(INERTIAL_DIR, SUBSET, label_map)
    video_samples = load_video_samples(KALMAN_CACHE, SUBSET, label_map)
    all_video_samples = video_samples  # Store all video samples for consistent stats computation
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

# ── Helper functions for dynamic routing and statistics verification ──────────────
def load_single_modality_checkpoint(model, checkpoint_path, modality, device):
    """Load single-modality checkpoint (sensor or video) into fusion backbone."""
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if modality == "video":
            model.video_backbone.load_state_dict(state_dict, strict=False)
        elif modality == "sensor":
            model.sensor_backbone.load_state_dict(state_dict, strict=False)
        return True
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}\n"
            f"Ensure {modality} models are trained first:\n"
            f"  python3 train_{modality}.py --config configs/{modality}.yaml"
        )

def verify_statistics_consistency(fold_stats_dict, test_subject, modality):
    """Verify that test statistics match training statistics (per-fold consistency)."""
    # For each fold, stats should be computed identically from same training subjects
    # This is a sanity check that normalization is consistent
    if modality == "sensor":
        mean = fold_stats_dict.get("mean", None)
        std = fold_stats_dict.get("std", None)
        if mean is not None and std is not None:
            return {"mean": float(np.mean(mean)), "std": float(np.mean(std))}
    elif modality == "video":
        mean = fold_stats_dict.get("mean", None)
        std = fold_stats_dict.get("std", None)
        if mean is not None and std is not None:
            return {"mean": float(np.mean(mean)), "std": float(np.mean(std))}
    return {}

def validate_checkpoint_format(state_dict, target_module, modality):
    """Validate that checkpoint keys are compatible with target module."""
    # Collect module parameter names
    module_keys = set()
    for name, param in target_module.named_parameters():
        module_keys.add(name)

    # Check for key mismatches (but strict=False handles this gracefully)
    checkpoint_keys = set(state_dict.keys())
    missing = module_keys - checkpoint_keys
    extra = checkpoint_keys - module_keys

    if missing:
        print(f"  ⚠️  {modality} checkpoint missing keys: {missing}")
    if extra:
        print(f"  ℹ️  {modality} checkpoint has extra keys (will be ignored): {extra}")

    return len(missing) < len(module_keys) * 0.5  # Warn if >50% of params missing

# ── Conformal Prediction Helper Functions ──────────────────────────────────────
def load_subject_data(subject: int, dataset_cls, fold_dataset_kwargs: dict):
    """Load all samples for a subject without DataLoader."""
    subject_samples = [(label, raw) for s, label, raw in samples if s == subject]
    dataset = dataset_cls(subject_samples, **fold_dataset_kwargs)

    x_list = []
    y_list = []
    for i in range(len(dataset)):
        if args.modality in ["fusion", "dynamic_fusion"]:
            (x_s, x_v), y = dataset[i]
            x_list.append((x_s, x_v))
        else:
            x, y = dataset[i]
            x_list.append(x)
        y_list.append(y)

    if args.modality in ["fusion", "dynamic_fusion"]:
        x_s_all = torch.stack([x[0] for x in x_list]).to(device)
        x_v_all = torch.stack([x[1] for x in x_list]).to(device)
        y_all = torch.tensor(y_list, dtype=torch.long).to(device)
        return (x_s_all, x_v_all), y_all
    else:
        x_all = torch.stack(x_list).to(device)
        y_all = torch.tensor(y_list, dtype=torch.long).to(device)
        return x_all, y_all

def get_softmax_scores(x, model, device):
    """Convert logits to softmax probabilities."""
    with torch.no_grad():
        if args.modality in ["fusion", "dynamic_fusion"]:
            if isinstance(x, tuple) and len(x) == 2:
                x_s, x_v = x
                if args.missing_sensor:
                    logits = model.video_backbone(x_v)
                elif args.missing_video:
                    logits = model.sensor_backbone(x_s)
                else:
                    logits = model(x_s, x_v)
            else:
                logits = model(x)
        else:
            logits = model(x)
        probs = F.softmax(logits, dim=1)
    return probs

def compute_nonconformity_scores(probs: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """Compute non-conformity scores: s_i = 1 - softmax[true_label]."""
    probs_np = probs.cpu().numpy()
    y_np = y.cpu().numpy()
    scores = 1.0 - probs_np[np.arange(len(y_np)), y_np]
    return scores

def compute_quantile_threshold(scores: np.ndarray, alpha: float = None) -> float:
    """Compute quantile threshold for conformal prediction."""
    if alpha is None:
        alpha = getattr(cfg, 'conformal', type('obj', (object,), {'alpha': 0.1})()).alpha
    n = len(scores)
    q_level = (n + 1) * (1 - alpha) / n
    q_hat = np.quantile(scores, q_level, method='higher')
    return q_hat

def build_prediction_set(probs: torch.Tensor, q_hat: float) -> list:
    """Build prediction sets using conformal prediction."""
    probs_np = probs.cpu().numpy()
    pred_sets = []
    empty_count = 0

    for prob_row in probs_np:
        pred_set = set(np.where(1.0 - prob_row <= q_hat)[0])

        if len(pred_set) == 0:
            pred_set = {np.argmax(prob_row)}
            empty_count += 1

        pred_sets.append(pred_set)

    if empty_count > 0:
        print(f"  ⚠️  {empty_count} empty prediction sets (defaulted to highest softmax class)")

    return pred_sets

def evaluate_conformal_metrics(y_true: np.ndarray, pred_sets: list) -> tuple:
    """Compute conformal prediction metrics."""
    coverage = sum(1 for y, pred_set in zip(y_true, pred_sets) if y in pred_set) / len(y_true)
    avg_set_size = np.mean([len(ps) for ps in pred_sets])
    ambiguity_rate = sum(1 for ps in pred_sets if len(ps) > 1) / len(pred_sets)
    return coverage, avg_set_size, ambiguity_rate

def plot_set_size_histogram(pred_sets_all_folds: list, num_classes: int, modality: str):
    """Plot histogram of prediction set sizes."""
    all_set_sizes = []
    for fold_sets in pred_sets_all_folds:
        all_set_sizes.extend([len(ps) for ps in fold_sets])

    fig, ax = plt.subplots(figsize=(10, 6))
    size_counts = {i: all_set_sizes.count(i) for i in range(1, num_classes + 1)}
    sizes = sorted(size_counts.keys())
    counts = [size_counts[s] for s in sizes]

    ax.bar(sizes, counts, color='steelblue', edgecolor='black')
    ax.set_xlabel('Prediction Set Size', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{modality.upper()} - Conformal Prediction Set Size Distribution', fontsize=14)
    ax.set_xticks(sizes)
    ax.grid(axis='y', alpha=0.3)

    return fig

# ── LOSO inference (one checkpoint per fold) ───────────────────────────────────
y_true_all: list[int] = []
y_pred_all: list[int] = []
fold_accuracies: list[float] = []
fold_f1_scores: list[float] = []
fold_statistics: dict = {}  # Track fold-specific statistics for verification
conformal_coverage_folds: list = [] if args.conformal else None
conformal_set_sizes_folds: list = [] if args.conformal else None
conformal_ambiguity_folds: list = [] if args.conformal else None
pred_sets_all_folds: list = [] if args.conformal else None

with torch.no_grad():
    for test_subject in subjects:
        # ── Load checkpoint(s) ──────────────────────────────────────────────────
        ckpt_file = fold_ckpt_path(test_subject)
        state = torch.load(ckpt_file, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()

        # For fusion modality with missing modalities: load single-modality checkpoints
        if args.modality == "fusion" and (args.missing_sensor or args.missing_video):
            if args.missing_sensor:
                video_ckpt = ROOT / "checkpoints/video" / f"fold_s{test_subject}.pt"
                load_single_modality_checkpoint(model, video_ckpt, "video", device)
            if args.missing_video:
                sensor_ckpt = ROOT / "checkpoints/sensor" / f"fold_s{test_subject}.pt"
                load_single_modality_checkpoint(model, sensor_ckpt, "sensor", device)

        # ── Compute fold-specific statistics ────────────────────────────────────
        fold_dataset_kwargs = dict(dataset_kwargs) if args.modality != "fusion" else {}
        fold_stats = {}

        if normalization_type == "global":
            train_subject_samples = [(s, y, data) for s, y, data in samples if s != test_subject]

            if args.modality == "sensor":
                sensor_mean, sensor_std = compute_global_stats(train_subject_samples)
                fold_stats["sensor"] = {"mean": sensor_mean, "std": sensor_std}
                fold_dataset_kwargs["global_stats"] = (sensor_mean, sensor_std)

            elif args.modality == "video":
                landmark_indices = get_landmark_indices(landmark_set)
                video_mean, video_std = compute_global_stats_video(train_subject_samples, landmark_indices)
                fold_stats["video"] = {"mean": video_mean, "std": video_std}
                fold_dataset_kwargs["global_stats"] = (video_mean, video_std)

            elif args.modality == "fusion":
                train_sensor_samples = [(s, y, data[0]) for s, y, data in train_subject_samples
                                       if isinstance(data, tuple) and len(data) == 2]

                # For video stats: use ALL video samples from training subjects (not just matched pairs)
                # This ensures video backbone receives data in same distribution as during training
                train_video_samples_all = [(s, y, raw) for s, y, raw in all_video_samples if s != test_subject]

                sensor_mean, sensor_std = compute_global_stats(train_sensor_samples)
                landmark_indices = get_landmark_indices(landmark_set)
                video_mean, video_std = compute_global_stats_video(train_video_samples_all, landmark_indices)

                fold_stats["sensor"] = {"mean": sensor_mean, "std": sensor_std}
                fold_stats["video"] = {"mean": video_mean, "std": video_std}

                fold_dataset_kwargs["global_stats_sensor"] = (sensor_mean, sensor_std)
                fold_dataset_kwargs["global_stats_video"] = (video_mean, video_std)
                fold_dataset_kwargs["sensor_max_len"] = sensor_max_len
                fold_dataset_kwargs["video_max_len"] = video_max_len
                fold_dataset_kwargs["feature_type"] = feature_type
                fold_dataset_kwargs["landmark_set"] = landmark_set
                fold_dataset_kwargs["normalization_type"] = normalization_type

        # Store for verification
        fold_statistics[test_subject] = fold_stats

        if args.conformal:
            # ══════════════════════════════════════════════════════════════════════
            # ── CONFORMAL PREDICTION EVALUATION ────────────────────────────────────
            # ══════════════════════════════════════════════════════════════════════

            # 1. Load calibration subject data
            fold_idx = list(subjects).index(test_subject)
            cal_subject = subjects[(fold_idx - 1) % len(subjects)]

            print(f"Fold {fold_idx + 1}: Test=s{test_subject}, Cal=s{cal_subject} (conformal)")

            # Compute calibration stats (same training subjects as test fold)
            if normalization_type == "global":
                train_subject_samples_cal = [(s, y, data) for s, y, data in samples if s != test_subject]
                # Reuse fold_dataset_kwargs which already has global stats
            else:
                train_subject_samples_cal = None

            # Load calibration data
            x_cal, y_cal = load_subject_data(cal_subject, dataset_cls, fold_dataset_kwargs)

            # Get softmax probabilities on calibration set
            probs_cal = get_softmax_scores(x_cal, model, device)

            # Compute non-conformity scores on calibration set
            scores_cal = compute_nonconformity_scores(probs_cal, y_cal)

            # Compute quantile threshold (90% confidence)
            q_hat = compute_quantile_threshold(scores_cal, alpha=0.1)

            # 2. Inference on test set
            x_test, y_test = load_subject_data(test_subject, dataset_cls, fold_dataset_kwargs)

            # Get softmax probabilities on test set
            probs_test = get_softmax_scores(x_test, model, device)

            # Build prediction sets
            pred_sets = build_prediction_set(probs_test, q_hat)

            # 3. Conformal metrics
            y_test_np = y_test.cpu().numpy()
            coverage, avg_set_size, ambiguity = evaluate_conformal_metrics(y_test_np, pred_sets)

            # Get top-1 predictions for confusion matrix
            top1_preds = np.array([max(pred_set, key=lambda c: probs_test[i, c].item())
                                   for i, pred_set in enumerate(pred_sets)])

            conformal_coverage_folds.append(coverage)
            conformal_set_sizes_folds.append(avg_set_size)
            conformal_ambiguity_folds.append(ambiguity)
            pred_sets_all_folds.append(pred_sets)

            print(f"  Quantile q_hat: {q_hat:.4f} | Coverage: {coverage:.3f} | Avg Set Size: {avg_set_size:.2f} | Ambiguity: {ambiguity:.3f}")

            # Pool for confusion matrix (using top-1 from prediction sets)
            y_true_all.extend(y_test_np)
            y_pred_all.extend(top1_preds)

            # Store fold metrics for standard evaluation
            fold_accuracies.append(accuracy_score(y_test_np, top1_preds))
            fold_f1_scores.append(f1_score(y_test_np, top1_preds, average="macro"))

        else:
            # ══════════════════════════════════════════════════════════════════════
            # ── STANDARD EVALUATION ────────────────────────────────────────────────
            # ══════════════════════════════════════════════════════════════════════

            # ── Create test dataset ────────────────────────────────────────────────
            test_samples = [(label, raw) for s, label, raw in samples if s == test_subject]
            test_data = dataset_cls(test_samples, **fold_dataset_kwargs)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

            y_fold_true: list[int] = []
            y_fold_pred: list[int] = []

            # ── Forward pass with dynamic routing ────────────────────────────────────
            for batch in test_loader:
                if args.modality in ["fusion", "dynamic_fusion"]:
                    (x_sensor, x_video), y = batch
                    x_sensor = x_sensor.to(device)
                    x_video = x_video.to(device)

                    if args.modality == "fusion":
                        # Dynamic routing: use single-modality backbones if available
                        if args.missing_sensor:
                            # Video only: forward through video backbone only
                            logits = model.video_backbone(x_video)
                        elif args.missing_video:
                            # Sensor only: forward through sensor backbone only
                            logits = model.sensor_backbone(x_sensor)
                        else:
                            # Bimodal: full fusion model
                            logits = model(x_sensor, x_video)
                else:
                    x, y = batch
                    x = x.to(device)
                    logits = model(x)

                preds = logits.argmax(dim=1).cpu().numpy()
                y_np = y.numpy()
                y_fold_true.extend(y_np)
                y_fold_pred.extend(preds)

            # ── Per-fold metrics ────────────────────────────────────────────────────
            fold_acc = accuracy_score(y_fold_true, y_fold_pred)
            fold_f1 = f1_score(y_fold_true, y_fold_pred, average="macro")
            fold_accuracies.append(fold_acc)
            fold_f1_scores.append(fold_f1)

            # Pool for confusion matrix
            y_true_all.extend(y_fold_true)
            y_pred_all.extend(y_fold_pred)

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# ── Verification: Statistics Consistency ────────────────────────────────────────

if fold_statistics:
    for modality in ["sensor", "video"]:
        fold_mean_values = []
        fold_std_values = []
        for subject, stats_dict in sorted(fold_statistics.items()):
            if modality in stats_dict:
                stats = stats_dict[modality]
                mean_arr = stats.get("mean")
                std_arr = stats.get("std")
                if mean_arr is not None and std_arr is not None:
                    fold_mean_values.append(np.mean(mean_arr))
                    fold_std_values.append(np.mean(std_arr))


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

# ── Conformal Prediction Summary ───────────────────────────────────────────────────
if args.conformal:
    conformal_alpha = getattr(cfg, 'conformal', type('obj', (object,), {'alpha': 0.1})()).alpha
    target_coverage = 1 - conformal_alpha

    coverage_mean = np.mean(conformal_coverage_folds)
    coverage_std = np.std(conformal_coverage_folds)
    set_size_mean = np.mean(conformal_set_sizes_folds)
    set_size_std = np.std(conformal_set_sizes_folds)
    ambiguity_mean = np.mean(conformal_ambiguity_folds)
    ambiguity_std = np.std(conformal_ambiguity_folds)

    print(f"\n{'='*70}")
    print(f"{'CONFORMAL PREDICTION SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"Alpha (α):           {conformal_alpha:.3f}")
    print(f"Target Coverage:     {target_coverage:.3f} ({int(target_coverage*100)}%)")
    print(f"Empirical Coverage:  {coverage_mean:.3f} ± {coverage_std:.3f}")
    print(f"Average Set Size:    {set_size_mean:.3f} ± {set_size_std:.3f} (out of {len(CLASS_NAMES)} classes)")
    print(f"Ambiguity Rate:      {ambiguity_mean:.3f} ± {ambiguity_std:.3f} (% with |C| > 1)")
    print(f"{'='*70}\n")

    # Generate set size histogram
    fig = plot_set_size_histogram(pred_sets_all_folds, len(CLASS_NAMES), args.modality.upper())
    hist_path = OUT_DIR / f"test_{args.modality}_conformal_set_size_histogram.png"
    display_and_save(fig, hist_path, title=f"Conformal set size histogram saved to {hist_path}")

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
            fold_dataset_kwargs = dict(dataset_kwargs) if args.modality not in ["fusion", "dynamic_fusion"] else {}
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

                    if args.missing_sensor:
                        logits = model.video_backbone(x_video)
                    elif args.missing_video:
                        logits = model.sensor_backbone(x_sensor)
                    else:
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
