"""
Train TransformerClassifier (sensor or video) or FusionTransformerClassifier with LOSO cross-validation.
Each fold is logged as a separate wandb run under the same group.

Usage:
    python train.py --modality sensor --config configs/sensor.yaml
    python train.py --modality video  --config configs/video.yaml
    python train.py --modality fusion --config configs/fusion.yaml
"""

import argparse
import math
import random
import sys
import warnings
from pathlib import Path
import numpy as np
from typing import Tuple, List

import matplotlib
matplotlib.use("Agg")
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from data.sensor_dataset import SensorDataset, load_sensor_samples, compute_global_stats
from data.video_dataset import PoseDataset, load_video_samples, compute_global_stats_video, get_landmark_indices
from data.fusion_dataset import FusionDataset, extract_matched_keys
from models.model import TransformerClassifier
from models.fusion import FusionTransformerClassifier
from utils import load_config
from video_pretraining import ensure_pose_cache

warnings.filterwarnings("ignore")


class TrainerModule(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        in_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        n_epochs: int,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.model = TransformerClassifier(
            n_classes=n_classes,
            in_dim=in_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        # Register class weights if provided
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Tuple, weighted: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y   = batch
        logits = self(x)
        weight = self.class_weights if (weighted and self.class_weights is not None) else None
        loss   = F.cross_entropy(logits, y, weight=weight)
        acc    = (logits.argmax(1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, weighted=True)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train/acc",  acc,  on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, weighted=False)
        self.log("val/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val/acc",  acc,  on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt   = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.n_epochs
        )
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


class FusionLightningModule(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        in_dim_sensor: int,
        in_dim_video: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        d_fusion: int,
        lr: float,
        weight_decay: float,
        n_epochs: int,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.model = FusionTransformerClassifier(
            n_classes=n_classes,
            in_dim_sensor=in_dim_sensor,
            in_dim_video=in_dim_video,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            d_fusion=d_fusion,
        )
        # Register class weights if provided
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, x_sensor: torch.Tensor, x_video: torch.Tensor) -> torch.Tensor:
        return self.model(x_sensor, x_video)

    def _shared_step(self, batch: Tuple, weighted: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        (x_sensor, x_video), y = batch
        logits = self(x_sensor, x_video)
        weight = self.class_weights if (weighted and self.class_weights is not None) else None
        loss   = F.cross_entropy(logits, y, weight=weight)
        acc    = (logits.argmax(1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, weighted=True)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train/acc",  acc,  on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, weighted=False)
        self.log("val/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val/acc",  acc,  on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt   = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.n_epochs
        )
        return [opt], [{"scheduler": sched, "interval": "epoch"}]
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["sensor", "video", "fusion"], required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-validation", action="store_true", help="Disable validation set (use training only)")
    parser.add_argument("--imbalance", action="store_true", help="Enable class imbalance mode (downsample training data)")
    parser.add_argument("--weighted_loss", action="store_true", help="Use weighted cross-entropy loss (requires --imbalance)")
    parser.add_argument("--augment_minority", action="store_true", help="Oversample and augment minority classes (requires --imbalance)")
    args = parser.parse_args()

    # Validate that --weighted_loss requires --imbalance
    if args.weighted_loss and not args.imbalance:
        raise ValueError("--weighted_loss requires --imbalance flag (WCE only meaningful with class imbalance)")
    
    if args.augment_minority and not args.imbalance:
        raise ValueError("--augment_minority requires --imbalance flag")

    if args.config is None:
        args.config = f"configs/{args.modality}.yaml"

    cfg = load_config(args.config)

    # Load imbalance config
    imbalance_target_actions = getattr(cfg, "imbalance_target_actions", [13, 22])
    imbalance_ratio = getattr(cfg, "imbalance_ratio", 0.5)

    if args.imbalance:
        print(f"\n⚠️  IMBALANCE MODE ENABLED")
        print(f"   Target actions: {imbalance_target_actions}")
        print(f"   Ratio: {imbalance_ratio} (keep {imbalance_ratio*100:.0f}%)")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed, workers=True)

    root    = Path(__file__).parent
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)

    label_map = {a: i for i, a in enumerate(cfg.subset)}

    def downsample_samples(samples: List[Tuple], target_actions: list, label_map: dict,
                          imbalance_ratio: float, seed: int) -> List[Tuple]:
        """Downsample training samples for target actions deterministically.

        Selects exactly ceil(count * imbalance_ratio) samples from each target action.
        Other actions are kept in full.

        Args:
            samples: list of (subject, label, data) tuples
            target_actions: list of action IDs to downsample (e.g., [13, 22])
            label_map: dict mapping action_id → class_index
            imbalance_ratio: float, keep this fraction of target action samples
            seed: int for reproducible random selection

        Returns:
            downsampled_samples: same format as input
        """
        random.seed(seed)
        target_labels = {label_map[a] for a in target_actions if a in label_map}

        # Group samples by label
        samples_by_label = {}
        for s, y, data in samples:
            if y not in samples_by_label:
                samples_by_label[y] = []
            samples_by_label[y].append((s, y, data))

        downsampled = []
        for label, label_samples in samples_by_label.items():
            if label in target_labels:
                # Downsample this target action
                target_count = max(1, math.ceil(len(label_samples) * imbalance_ratio))
                selected = random.sample(label_samples, target_count)
                downsampled.extend(selected)
            else:
                # Keep all non-target samples
                downsampled.extend(label_samples)

        return downsampled

    def oversample_to_parity(samples: List[Tuple], target_actions: list, label_map: dict,
                            seed: int) -> List[Tuple]:
        """Oversample minority classes to match the maximum class count (parity).

        Args:
            samples: list of (subject, label, data) tuples after downsampling
            target_actions: list of action IDs to oversample (e.g., [13, 22])
            label_map: dict mapping action_id → class_index
            seed: int for reproducible random selection of which samples to duplicate

        Returns:
            oversampled_samples: list with minority classes duplicated to parity
        """
        random.seed(seed)
        target_labels = {label_map[a] for a in target_actions if a in label_map}

        # Group samples by label
        samples_by_label = {}
        for s, y, data in samples:
            if y not in samples_by_label:
                samples_by_label[y] = []
            samples_by_label[y].append((s, y, data))

        # Find maximum count across all classes
        max_count = max(len(label_samples) for label_samples in samples_by_label.values())

        # Oversample target labels to match max_count
        oversampled = []
        for label, label_samples in samples_by_label.items():
            oversampled.extend(label_samples)
            if label in target_labels:
                # Duplicate samples until we reach max_count
                current_count = len(label_samples)
                needed = max_count - current_count
                if needed > 0:
                    duplicates = random.choices(label_samples, k=needed)
                    oversampled.extend(duplicates)

        return oversampled


    def compute_class_weights(train_subject_samples: List[Tuple], n_classes: int) -> torch.Tensor:
        """Compute per-class weights for weighted cross-entropy loss.

        Weight formula: weight_i = total_samples / (n_classes * count_i)
        This gives rarer classes higher weights.
        """
        # Count samples per class
        class_counts = [0] * n_classes
        for s, y, data in train_subject_samples:
            class_counts[y] += 1

        total_samples = len(train_subject_samples)

        # Compute weights: avoid division by zero
        weights = []
        for count in class_counts:
            if count == 0:
                weight = 1.0
            else:
                weight = total_samples / (n_classes * count)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)

    # Modality-specific setup
    if args.modality == "sensor":
        samples = load_sensor_samples(root / cfg.inertial_dir, cfg.subset, label_map)
        in_dim  = cfg.in_dim
        dataset_cls = SensorDataset
        feature_type = getattr(cfg, "feature_type", "raw+velocity")
        normalization_type = getattr(cfg, "normalization_type", "per_sample")
        dataset_kwargs = {"max_len": cfg.max_len, "feature_type": feature_type,
                         "normalization_type": normalization_type}

    elif args.modality == "video":
        kalman_cache = root / cfg.kalman_cache
        ensure_pose_cache(root, cfg, cfg.subset, label_map)

        samples = load_video_samples(kalman_cache, cfg.subset, label_map)
        in_dim = cfg.in_dim
        dataset_cls = PoseDataset
        landmark_set = getattr(cfg, "landmark_set", "all")
        normalization_type = getattr(cfg, "normalization_type", "per_sample")
        dataset_kwargs = {"max_len": cfg.max_len, "landmark_set": landmark_set,
                         "normalization_type": normalization_type}

    elif args.modality == "fusion":
        sensor_samples = load_sensor_samples(root / cfg.inertial_dir, cfg.subset, label_map)
        kalman_cache = root / cfg.kalman_cache
        ensure_pose_cache(root, cfg, cfg.subset, label_map)
        video_samples = load_video_samples(kalman_cache, cfg.subset, label_map)
        matched_samples = extract_matched_keys(sensor_samples, video_samples)
        samples = matched_samples
        dataset_cls = FusionDataset
        normalization_type = getattr(cfg, "normalization_type", "per_sample")
        sensor_max_len = getattr(cfg, "max_len_sensor", 256)
        video_max_len = getattr(cfg, "max_len_video", 128)
        feature_type = getattr(cfg, "feature_type", "raw+velocity")
        landmark_set = getattr(cfg, "landmark_set", "hands_legs_hips")
        dataset_kwargs = {
            "sensor_max_len": sensor_max_len,
            "video_max_len": video_max_len,
            "feature_type": feature_type,
            "landmark_set": landmark_set,
            "normalization_type": normalization_type,
        }

    else:
        raise ValueError(f"Unknown modality: {args.modality}")

    subjects = sorted(set(s for s, _, _ in samples))

    if len(samples) == 0:
        print(f"\n❌ No {args.modality} samples found")
        if args.modality == "video":
            print(f"   Run video_pretraining.py first to generate Kalman pose cache")
        elif args.modality == "fusion":
            print(f"   Ensure both sensor and video data are available and matched")
        sys.exit(1)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else "cpu"
    )

    if args.augment_minority and args.imbalance:
        ckpt_root = root / f"checkpoints/imbalance_aug/{args.modality}"
    elif args.weighted_loss and args.imbalance:
        ckpt_root = root / f"checkpoints/imbalance_weighted/{args.modality}"
    elif args.imbalance:
        ckpt_root = root / f"checkpoints/imbalance/{args.modality}"
    else:
        ckpt_root = root / cfg.checkpoint_dir
    ckpt_root.mkdir(parents=True, exist_ok=True)

    for fold_idx, test_subject in enumerate(subjects):
        if args.no_validation:
            train_subject_samples = [(s, y, data) for s, y, data in samples if s != test_subject]
            train_samples = [(y, data) for s, y, data in train_subject_samples]
        else:
            val_subject = subjects[(fold_idx - 1) % len(subjects)]
            train_subject_samples = [(s, y, data) for s, y, data in samples if s != test_subject and s != val_subject]
            train_samples = [(y, data) for s, y, data in train_subject_samples]
            val_samples = [(y, data) for s, y, data in samples if s == val_subject]

        # Apply imbalance downsampling to training data only (not val/test)
        if args.imbalance:
            original_count = len(train_subject_samples)
            train_subject_samples = downsample_samples(
                train_subject_samples,
                imbalance_target_actions,
                label_map,
                imbalance_ratio,
                cfg.seed + fold_idx
            )
            train_samples = [(y, data) for s, y, data in train_subject_samples]
            print(f"Fold {fold_idx + 1} (test=s{test_subject}): {original_count} → {len(train_subject_samples)} samples")

        # Apply oversampling if --augment_minority enabled
        if args.augment_minority:
            original_downsampled = len(train_subject_samples)
            train_subject_samples = oversample_to_parity(
                train_subject_samples,
                imbalance_target_actions,
                label_map,
                cfg.seed + fold_idx
            )
            train_samples = [(y, data) for s, y, data in train_subject_samples]
            print(f"  After oversampling: {len(train_subject_samples)} samples (parity achieved)")

        # Compute class weights from downsampled training data if WCE enabled
        class_weights = None
        if args.weighted_loss:
            class_weights = compute_class_weights(train_subject_samples, len(cfg.subset))
            # Move to appropriate device IMMEDIATELY to avoid initialization errors
            class_weights = class_weights.to(device)
            
            # Log weight values for minority classes (13, 22) in fold 0 for verification
            if fold_idx == 0:
                target_label_13 = label_map.get(13)
                target_label_22 = label_map.get(22)
                print(f"  Class weights computed (fold {fold_idx + 1}):")
                if target_label_13 is not None:
                    print(f"    Action 13 (label {target_label_13}): {class_weights[target_label_13]:.4f}")
                if target_label_22 is not None:
                    print(f"    Action 22 (label {target_label_22}): {class_weights[target_label_22]:.4f}")


        if len(train_samples) == 0:
            print(f"\n❌ No training samples for fold s{test_subject}")
            print(f"   Total samples: {len(samples)} | Subjects: {subjects}")
            print(f"   Test: s{test_subject} | Val: s{val_subject if not args.no_validation else 'N/A'}")
            sys.exit(1)

        fold_dataset_kwargs = dict(dataset_kwargs)

        # Setup augmentation parameters
        augment_minority = args.augment_minority and args.imbalance
        minority_classes = set()
        augment_params = {}
        if augment_minority:
            minority_classes = {label_map[a] for a in imbalance_target_actions if a in label_map}
            aug_ns = getattr(cfg, "augmentation", None)
            augment_params = vars(aug_ns) if aug_ns is not None else {}
        fold_dataset_kwargs["augment_minority"] = augment_minority
        fold_dataset_kwargs["minority_classes"] = minority_classes
        fold_dataset_kwargs["augment_minority_params"] = augment_params

        if args.modality == "fusion":
            if normalization_type == "global":
                train_sensor_samples = [(s, y, data[0]) for s, y, data in train_subject_samples
                                       if isinstance(data, tuple) and len(data) == 2]
                train_video_samples = [(s, y, data[1]) for s, y, data in train_subject_samples
                                      if isinstance(data, tuple) and len(data) == 2]
                fold_dataset_kwargs["global_stats_sensor"] = compute_global_stats(train_sensor_samples)
                landmark_indices = get_landmark_indices(landmark_set)
                fold_dataset_kwargs["global_stats_video"] = compute_global_stats_video(train_video_samples, landmark_indices)
        else:
            if normalization_type == "global":
                if args.modality == "sensor":
                    fold_dataset_kwargs["global_stats"] = compute_global_stats(train_subject_samples)
                else:
                    landmark_indices = get_landmark_indices(landmark_set)
                    fold_dataset_kwargs["global_stats"] = compute_global_stats_video(train_subject_samples, landmark_indices)

        train_loader = DataLoader(
            dataset_cls(train_samples, augment=True, **fold_dataset_kwargs),
            batch_size=cfg.training.batch_size, shuffle=True, num_workers=0,
        )

        if not args.no_validation:
            val_loader = DataLoader(
                dataset_cls(val_samples, augment=False, **fold_dataset_kwargs),
                batch_size=cfg.training.batch_size, shuffle=False, num_workers=0,
            )

        # Compute WandB group suffix
        wandb_suffix = ""
        if args.augment_minority and args.imbalance:
            wandb_suffix = "_imbalance_aug"
        elif args.weighted_loss and args.imbalance:
            wandb_suffix = "_imbalance_wce"
        elif args.imbalance:
            wandb_suffix = "_imbalance"
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            group=cfg.wandb.group + wandb_suffix,
            name=f"fold_s{test_subject}",
            log_model=False,
            settings=wandb.Settings(quiet=True)
        )

        torch.manual_seed(cfg.seed)

        if args.modality == "fusion":
            module = FusionLightningModule(
                n_classes=len(cfg.subset),
                in_dim_sensor=cfg.in_dim_sensor,
                in_dim_video=cfg.in_dim_video,
                d_model=cfg.model.d_model,
                n_heads=cfg.model.n_heads,
                n_layers=cfg.model.n_layers,
                dropout=cfg.model.dropout,
                d_fusion=cfg.model.d_fusion,
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
                n_epochs=cfg.training.n_epochs,
                class_weights=class_weights,
            )
        else:
            module = TrainerModule(
                n_classes=len(cfg.subset),
                in_dim=cfg.in_dim,
                d_model=cfg.model.d_model,
                n_heads=cfg.model.n_heads,
                n_layers=cfg.model.n_layers,
                dropout=cfg.model.dropout,
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
                n_epochs=cfg.training.n_epochs,
                class_weights=class_weights,
            )

        monitor_metric = "val/loss" if not args.no_validation else "train/loss"

        trainer = pl.Trainer(
            max_epochs=cfg.training.n_epochs,
            logger=wandb_logger,
            callbacks=[
                ModelCheckpoint(
                    dirpath=str(ckpt_root / f"fold_s{test_subject}"),
                    filename="best",
                    save_top_k=1,
                    monitor=monitor_metric,
                    mode="min",
                )
            ],
            gradient_clip_val=cfg.training.grad_clip,
            accelerator="auto",
            enable_progress_bar=True,
            enable_model_summary=(fold_idx == 0),
        )

        if args.no_validation:
            trainer.fit(module, train_loader)
        else:
            trainer.fit(module, train_loader, val_loader)

        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        if best_ckpt_path:
            ckpt = torch.load(best_ckpt_path, map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            pt_path = ckpt_root / f"fold_s{test_subject}.pt"
            torch.save(state_dict, pt_path)
            print(f"✓ Saved best checkpoint to {pt_path}")

        wandb.finish()


SensorLightningModule = TrainerModule
VideoLightningModule = TrainerModule


if __name__ == "__main__":
    main()
