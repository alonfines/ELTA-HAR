"""
Train TransformerClassifier (sensor or video) or FusionTransformerClassifier with LOSO cross-validation.
Each fold is logged as a separate wandb run under the same group.

Usage:
    python train.py --modality sensor --config configs/sensor.yaml
    python train.py --modality video  --config configs/video.yaml
    python train.py --modality fusion --config configs/fusion.yaml
"""

import argparse
import random
import sys
import warnings
from pathlib import Path
import numpy as np
from typing import Tuple

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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerClassifier(
            n_classes=n_classes,
            in_dim=in_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        acc    = (logits.argmax(1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train/acc",  acc,  on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
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
    ):
        super().__init__()
        self.save_hyperparameters()
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

    def forward(self, x_sensor: torch.Tensor, x_video: torch.Tensor) -> torch.Tensor:
        return self.model(x_sensor, x_video)

    def _shared_step(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        (x_sensor, x_video), y = batch
        logits = self(x_sensor, x_video)
        loss   = F.cross_entropy(logits, y)
        acc    = (logits.argmax(1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train/acc",  acc,  on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
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
    args = parser.parse_args()

    if args.config is None:
        args.config = f"configs/{args.modality}.yaml"

    cfg = load_config(args.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed, workers=True)

    root    = Path(__file__).parent
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)

    label_map = {a: i for i, a in enumerate(cfg.subset)}

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

        if len(train_samples) == 0:
            print(f"\n❌ No training samples for fold s{test_subject}")
            print(f"   Total samples: {len(samples)} | Subjects: {subjects}")
            print(f"   Test: s{test_subject} | Val: s{val_subject if not args.no_validation else 'N/A'}")
            sys.exit(1)

        fold_dataset_kwargs = dict(dataset_kwargs)

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

        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            group=cfg.wandb.group,
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
            )

        monitor_metric = "val/loss" if not args.no_validation else "train/loss"

        trainer = pl.Trainer(
            max_epochs=cfg.training.n_epochs,
            logger=wandb_logger,
            callbacks=[
                ModelCheckpoint(
                    dirpath=str(ckpt_root / f"{args.modality}_fold_s{test_subject}"),
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
            pt_path = ckpt_root / f"{args.modality}_fold_s{test_subject}.pt"
            torch.save(state_dict, pt_path)
            print(f"✓ Saved best checkpoint to {pt_path}")

        wandb.finish()


SensorLightningModule = TrainerModule
VideoLightningModule = TrainerModule


if __name__ == "__main__":
    main()
