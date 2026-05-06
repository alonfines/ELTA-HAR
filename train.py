"""
Train TransformerClassifier (sensor or video) or FusionTransformerClassifier with LOSO cross-validation.
Each fold is logged as a separate wandb run under the same group.

Usage:
    python train.py --modality sensor --config configs/sensor.yaml --conformal
    python train.py --modality fusion --imbalance --augment_minority --conformal
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
    parser.add_argument("--conformal", action="store_true", help="Enable split-conformal prediction (6-1-1 split)")
    args = parser.parse_args()

    if args.weighted_loss and not args.imbalance:
        raise ValueError("--weighted_loss requires --imbalance flag")
    
    if args.augment_minority and not args.imbalance:
        raise ValueError("--augment_minority requires --imbalance flag")
    
    if args.conformal:
        args.no_validation = True  # Forced for virgin calibration split

    if args.config is None:
        args.config = f"configs/{args.modality}.yaml"

    cfg = load_config(args.config)

    imbalance_target_actions = getattr(cfg, "imbalance_target_actions", [13, 22])
    imbalance_ratio = getattr(cfg, "imbalance_ratio", 0.5)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed, workers=True)

    root = Path(__file__).parent
    label_map = {a: i for i, a in enumerate(cfg.subset)}

    # Helper functions for downsampling and oversampling
    def downsample_samples(samples, target_actions, label_map, ratio, seed):
        random.seed(seed)
        target_labels = {label_map[a] for a in target_actions if a in label_map}
        samples_by_label = {}
        for s, y, data in samples:
            samples_by_label.setdefault(y, []).append((s, y, data))
        downsampled = []
        for label, label_samples in samples_by_label.items():
            if label in target_labels:
                target_count = max(1, math.ceil(len(label_samples) * ratio))
                downsampled.extend(random.sample(label_samples, target_count))
            else:
                downsampled.extend(label_samples)
        return downsampled

    def oversample_to_parity(samples, target_actions, label_map, seed):
        random.seed(seed)
        target_labels = {label_map[a] for a in target_actions if a in label_map}
        samples_by_label = {}
        for s, y, data in samples:
            samples_by_label.setdefault(y, []).append((s, y, data))
        max_count = max(len(ls) for ls in samples_by_label.values())
        oversampled = []
        for label, label_samples in samples_by_label.items():
            oversampled.extend(label_samples)
            if label in target_labels:
                needed = max_count - len(label_samples)
                if needed > 0:
                    oversampled.extend(random.choices(label_samples, k=needed))
        return oversampled

    def compute_class_weights(train_subject_samples, n_classes):
        class_counts = [0] * n_classes
        for s, y, data in train_subject_samples:
            class_counts[y] += 1
        total = len(train_subject_samples)
        return torch.tensor([total/(n_classes*c) if c > 0 else 1.0 for c in class_counts], dtype=torch.float32)

    # Modality Data Loading
    if args.modality == "sensor":
        samples = load_sensor_samples(root / cfg.inertial_dir, cfg.subset, label_map)
        dataset_cls = SensorDataset
        dataset_kwargs = {"max_len": cfg.max_len, "feature_type": getattr(cfg, "feature_type", "raw+velocity"),
                         "normalization_type": getattr(cfg, "normalization_type", "per_sample")}
    elif args.modality == "video":
        kalman_cache = root / cfg.kalman_cache
        ensure_pose_cache(root, cfg, cfg.subset, label_map)
        samples = load_video_samples(kalman_cache, cfg.subset, label_map)
        dataset_cls = PoseDataset
        dataset_kwargs = {"max_len": cfg.max_len, "landmark_set": getattr(cfg, "landmark_set", "all"),
                         "normalization_type": getattr(cfg, "normalization_type", "per_sample")}
    elif args.modality == "fusion":
        sensor_samples = load_sensor_samples(root / cfg.inertial_dir, cfg.subset, label_map)
        kalman_cache = root / cfg.kalman_cache
        ensure_pose_cache(root, cfg, cfg.subset, label_map)
        video_samples = load_video_samples(kalman_cache, cfg.subset, label_map)
        samples = extract_matched_keys(sensor_samples, video_samples)
        dataset_cls = FusionDataset
        dataset_kwargs = {
            "sensor_max_len": getattr(cfg, "max_len_sensor", 256),
            "video_max_len": getattr(cfg, "max_len_video", 128),
            "feature_type": getattr(cfg, "feature_type", "raw+velocity"),
            "landmark_set": getattr(cfg, "landmark_set", "hands_legs_hips"),
            "normalization_type": getattr(cfg, "normalization_type", "per_sample"),
        }

    subjects = sorted(set(s for s, _, _ in samples))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint Routing
    if args.conformal:
        ckpt_root = root / f"checkpoints/conformal/{args.modality}"
    elif args.augment_minority and args.imbalance:
        ckpt_root = root / f"checkpoints/imbalance_aug/{args.modality}"
    elif args.weighted_loss and args.imbalance:
        ckpt_root = root / f"checkpoints/imbalance_weighted/{args.modality}"
    elif args.imbalance:
        ckpt_root = root / f"checkpoints/imbalance/{args.modality}"
    else:
        ckpt_root = root / cfg.checkpoint_dir
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # LOSO Loop
    for fold_idx, test_subject in enumerate(subjects):
        if args.conformal:
            cal_subject = subjects[(fold_idx - 1) % len(subjects)]
            train_subject_samples = [(s, y, data) for s, y, data in samples if s != test_subject and s != cal_subject]
            print(f"Fold {fold_idx + 1} [CONFORMAL]: Test=s{test_subject}, Cal=s{cal_subject}, Train=6 subjects")
        elif args.no_validation:
            train_subject_samples = [(s, y, data) for s, y, data in samples if s != test_subject]
        else:
            val_subject = subjects[(fold_idx - 1) % len(subjects)]
            train_subject_samples = [(s, y, data) for s, y, data in samples if s != test_subject and s != val_subject]
            val_samples = [(y, data) for s, y, data in samples if s == val_subject]

        # Imbalance Handling
        if args.imbalance:
            train_subject_samples = downsample_samples(train_subject_samples, imbalance_target_actions, label_map, imbalance_ratio, cfg.seed + fold_idx)
            if args.augment_minority:
                train_subject_samples = oversample_to_parity(train_subject_samples, imbalance_target_actions, label_map, cfg.seed + fold_idx)

        train_samples = [(y, data) for s, y, data in train_subject_samples]
        class_weights = compute_class_weights(train_subject_samples, len(cfg.subset)).to(device) if args.weighted_loss else None

        # Dataset Setup with Global Stats (if needed)
        fold_kwargs = dict(dataset_kwargs)
        norm_type = fold_kwargs.get("normalization_type", "per_sample")

        if norm_type == "global":
            if args.modality == "fusion":
                train_sensor = [(s, y, d[0]) for s, y, d in train_subject_samples if isinstance(d, tuple) and len(d) == 2]
                train_video = [(s, y, d[1]) for s, y, d in train_subject_samples if isinstance(d, tuple) and len(d) == 2]
                if train_sensor:
                    fold_kwargs["global_stats_sensor"] = compute_global_stats(train_sensor)
                if train_video:
                    fold_kwargs["global_stats_video"] = compute_global_stats_video(
                        train_video,
                        get_landmark_indices(fold_kwargs["landmark_set"])
                    )
                print(f"  Computed global stats for fusion (sensor: {len(train_sensor)}, video: {len(train_video)} samples)")
            elif args.modality == "sensor":
                fold_kwargs["global_stats"] = compute_global_stats(train_subject_samples)
                print(f"  Computed global stats for sensor ({len(train_subject_samples)} samples)")
            else:  # video
                fold_kwargs["global_stats"] = compute_global_stats_video(
                    train_subject_samples,
                    get_landmark_indices(fold_kwargs["landmark_set"])
                )
                print(f"  Computed global stats for video ({len(train_subject_samples)} samples)")

        if args.augment_minority and args.imbalance:
            fold_kwargs.update({
                "augment_minority": True,
                "minority_classes": {label_map[a] for a in imbalance_target_actions if a in label_map},
                "augment_minority_params": vars(cfg.augmentation) if hasattr(cfg, "augmentation") else {}
            })

        train_loader = DataLoader(dataset_cls(train_samples, augment=True, **fold_kwargs), 
                                  batch_size=cfg.training.batch_size, shuffle=True)
        val_loader = DataLoader(dataset_cls(val_samples, augment=False, **fold_kwargs), 
                                batch_size=cfg.training.batch_size) if not args.no_validation else None

        # Logging
        wandb_suffix = "_conformal" if args.conformal else ("_imbalance_aug" if args.augment_minority else "_imbalance_wce" if args.weighted_loss else "_imbalance" if args.imbalance else "")
        wandb_logger = WandbLogger(project=cfg.wandb.project, group=cfg.wandb.group + wandb_suffix, name=f"fold_s{test_subject}")

        # Module Initialization
        if args.modality == "fusion":
            module = FusionLightningModule(n_classes=len(cfg.subset), in_dim_sensor=cfg.in_dim_sensor, in_dim_video=cfg.in_dim_video,
                                          d_model=cfg.model.d_model, n_heads=cfg.model.n_heads, n_layers=cfg.model.n_layers,
                                          dropout=cfg.model.dropout, d_fusion=cfg.model.d_fusion, lr=cfg.training.lr, 
                                          weight_decay=cfg.training.weight_decay, n_epochs=cfg.training.n_epochs, class_weights=class_weights)
        else:
            module = TrainerModule(n_classes=len(cfg.subset), in_dim=cfg.in_dim, d_model=cfg.model.d_model, n_heads=cfg.model.n_heads, 
                                   n_layers=cfg.model.n_layers, dropout=cfg.model.dropout, lr=cfg.training.lr, 
                                   weight_decay=cfg.training.weight_decay, n_epochs=cfg.training.n_epochs, class_weights=class_weights)

        trainer = pl.Trainer(
            max_epochs=cfg.training.n_epochs, logger=wandb_logger,
            callbacks=[ModelCheckpoint(dirpath=str(ckpt_root / f"fold_s{test_subject}"), filename="best", 
                                      monitor="val/loss" if not args.no_validation else "train/loss", mode="min")],
            gradient_clip_val=cfg.training.grad_clip, accelerator="auto"
        )

        trainer.fit(module, train_loader, val_loader)

        # Export weights
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            ckpt = torch.load(best_path, map_location="cpu")
            torch.save({k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}, ckpt_root / f"fold_s{test_subject}.pt")
        
        wandb.finish()

if __name__ == "__main__":
    main()