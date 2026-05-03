"""
Part 5 — Class Imbalance: Detect, Introduce, Fix.

Workflow:
  1. Detect: show the natural (balanced) class distribution
  2. Introduce: keep only 20% of samples from 3 minority action classes → ~4.5:1 ratio
  3. Fix with two techniques:
       Weighted CE loss  — inverse-frequency loss weights, no data change
       Targeted augmentation — time warp + amplitude warp on minority classes only
  4. Compare four variants via LOSO: balanced / baseline / weighted / augmented
     Primary metric: Macro F1 (accuracy is misleading under imbalance)

Usage:
    python part5_imbalance.py
"""

import random
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from data.sensor_dataset import SensorDataset, add_velocity, load_sensor_samples, pad_or_truncate
from data.augmentation import time_warp, amplitude_warp
from models.model import TransformerClassifier
from utils import load_action_names, save_confusion_matrix

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
SUBSET        = [1, 2, 4, 13, 19, 22, 23, 27]
LABEL_MAP     = {a: i for i, a in enumerate(SUBSET)}
# Actions 1, 2, 4 → label indices 0, 1, 2 (fine-motor wrist gestures)
MINORITY_ACTS = {1, 2, 4}
MINORITY_IDX  = {LABEL_MAP[a] for a in MINORITY_ACTS}   # {0, 1, 2}
KEEP_FRAC     = 0.20    # keep 20% of minority samples → ~4.5:1 ratio
N_AUG         = 4       # augmented copies per minority original

N_EPOCHS  = 30
BATCH     = 16
LR        = 1e-3
WD        = 5e-3
MAX_LEN   = 256
SEED      = 42


def augment_strong(x: np.ndarray) -> np.ndarray:
    """Time warp + amplitude warp + Gaussian jitter for minority-class samples."""
    x = time_warp(x)
    x = amplitude_warp(x)
    x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
    return x


# ── Targeted-augmentation dataset ─────────────────────────────────────────────

class TargetedAugDataset(Dataset):
    """
    For majority-class samples: keep original + standard augmentation.
    For minority-class samples: keep original + N_AUG strong-augmented copies.
    This rebalances counts without duplicating exact sequences.
    """

    def __init__(
        self,
        samples: List[Tuple[int, np.ndarray]],
        minority_idx: set,
        max_len: int,
        n_aug: int = N_AUG,
    ):
        # Each entry: (label, raw, use_strong_aug)
        expanded: List[Tuple[int, np.ndarray, bool]] = []
        for y, raw in samples:
            expanded.append((y, raw, False))          # original (standard aug at __getitem__)
            if y in minority_idx:
                for _ in range(n_aug):
                    expanded.append((y, raw, True))   # strongly augmented copy

        rng = random.Random(SEED)
        rng.shuffle(expanded)
        self._data   = expanded
        self._max    = max_len

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        y, raw, strong = self._data[idx]
        x = pad_or_truncate(raw, self._max).astype(np.float32)

        if strong:
            x = augment_strong(x)
        else:
            # Standard augmentation identical to SensorDataset
            x += np.random.normal(0, 0.01, x.shape).astype(np.float32)
            x *= np.random.uniform(0.9, 1.1)

        x   = add_velocity(x)
        mu  = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-8
        return torch.from_numpy((x - mu) / std), y


# ── Lightning module ───────────────────────────────────────────────────────────

class SensorModule(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = TransformerClassifier(n_classes, in_dim=12, d_model=64,
                                           n_heads=4, n_layers=2, dropout=0.3)
        ones = torch.ones(n_classes)
        self.register_buffer("class_weights", class_weights if class_weights is not None else ones)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, _):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y, weight=self.class_weights.to(x.device))
        acc    = (logits.argmax(1) == y).float().mean()
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train/acc",  acc,  on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt   = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


# ── Helpers ────────────────────────────────────────────────────────────────────

def introduce_imbalance(
    samples: List[Tuple[int, int, np.ndarray]],
    minority_acts: set,
    keep_frac: float,
    seed: int = SEED,
) -> List[Tuple[int, int, np.ndarray]]:
    rng = random.Random(seed)
    return [
        (s, y, raw) for s, y, raw in samples
        if SUBSET[y] not in minority_acts or rng.random() < keep_frac
    ]


def compute_class_weights(
    train_samples: List[Tuple[int, np.ndarray]],
    n_classes: int,
) -> torch.Tensor:
    counts = np.bincount([y for y, _ in train_samples], minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w = len(train_samples) / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    module: SensorModule,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    module.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        preds = module(x.to(device)).argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())
    return np.array(y_true), np.array(y_pred)


# ── LOSO training ─────────────────────────────────────────────────────────────

def run_loso(
    samples: List[Tuple[int, int, np.ndarray]],
    n_classes: int,
    variant: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    variant: "balanced" | "baseline" | "weighted" | "augmented"
    Returns aggregated (y_true, y_pred) across all LOSO folds.
    """
    subjects = sorted(set(s for s, _, _ in samples))
    all_true, all_pred = [], []

    for test_subj in subjects:
        train_all = [(y, raw) for s, y, raw in samples if s != test_subj]
        test_all  = [(y, raw) for s, y, raw in samples if s == test_subj]

        weights = compute_class_weights(train_all, n_classes) if variant == "weighted" else None

        if variant == "augmented":
            train_ds = TargetedAugDataset(train_all, MINORITY_IDX, MAX_LEN)
        else:
            train_ds = SensorDataset(train_all, max_len=MAX_LEN, augment=True)

        test_ds = SensorDataset(test_all, max_len=MAX_LEN, augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

        torch.manual_seed(SEED)
        module = SensorModule(n_classes=n_classes, class_weights=weights)

        trainer = pl.Trainer(
            max_epochs=N_EPOCHS,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="auto",
            gradient_clip_val=1.0,
        )
        trainer.fit(module, train_loader)

        y_true, y_pred = evaluate(module.to(device), test_loader, device)
        all_true.extend(y_true)
        all_pred.extend(y_pred)

    return np.array(all_true), np.array(all_pred)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_distributions(
    natural: List[Tuple[int, int, np.ndarray]],
    imbalanced: List[Tuple[int, int, np.ndarray]],
    class_names: List[str],
    path: Path,
) -> None:
    n   = len(class_names)
    nat = Counter(y for _, y, _ in natural)
    imb = Counter(y for _, y, _ in imbalanced)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    colors_nat = ["steelblue"] * n
    colors_imb = [
        "#d9534f" if i in MINORITY_IDX else "steelblue" for i in range(n)
    ]

    for ax, counts, colors, title in [
        (axes[0], nat, colors_nat, "Natural distribution (balanced)"),
        (axes[1], imb, colors_imb, f"After imbalance (minority at {int(KEEP_FRAC*100)}%)"),
    ]:
        vals = [counts.get(i, 0) for i in range(n)]
        bars = ax.bar(class_names, vals, color=colors)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Sample count")
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, max(nat.values()) * 1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                    str(v), ha="center", va="bottom", fontsize=8)

    axes[1].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#d9534f", label="minority (kept 20%)"),
            plt.Rectangle((0, 0), 1, 1, color="steelblue", label="majority"),
        ],
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_summary(results: dict, class_names: List[str], path: Path) -> None:
    variants = ["balanced", "baseline", "weighted", "augmented"]
    labels   = ["Balanced\n(reference)", "Baseline\n(no fix)", "Weighted\nCE loss", "Targeted\naugmentation"]
    colors   = ["#2ca02c", "#d9534f", "#5bc0de", "#f0ad4e"]

    macro_f1     = [f1_score(*results[v], average="macro") for v in variants]
    per_cls_f1   = {v: f1_score(*results[v], average=None)  for v in variants}
    minority_f1  = {v: per_cls_f1[v][list(MINORITY_IDX)].mean() for v in variants}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Macro F1 bar ──────────────────────────────────────────────────────────
    bars = axes[0].bar(labels, macro_f1, color=colors, width=0.5)
    axes[0].axhline(macro_f1[0], color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.6)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("LOSO Macro F1", fontweight="bold")
    axes[0].set_ylabel("Macro F1")
    for bar, v in zip(bars, macro_f1):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # ── Minority-class F1 bar ─────────────────────────────────────────────────
    min_f1_vals = [minority_f1[v] for v in variants]
    bars2 = axes[1].bar(labels, min_f1_vals, color=colors, width=0.5)
    axes[1].axhline(min_f1_vals[0], color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.6)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("Mean F1 — minority classes only\n(actions 1, 2, 4)", fontweight="bold")
    axes[1].set_ylabel("F1")
    for bar, v in zip(bars2, min_f1_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # ── Per-class F1 grouped bar ──────────────────────────────────────────────
    n_cls = len(class_names)
    x     = np.arange(n_cls)
    w     = 0.18
    for i, (v, col, lbl) in enumerate(zip(variants, colors, labels)):
        offset = (i - 1.5) * w
        axes[2].bar(x + offset, per_cls_f1[v], w,
                    label=lbl.replace("\n", " "), color=col, alpha=0.85)
    # Shade minority columns
    for mi in MINORITY_IDX:
        axes[2].axvspan(mi - 0.5, mi + 0.5, color="#ffcccc", alpha=0.25, zorder=0)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_title("Per-class F1 by variant\n(pink = minority classes)", fontweight="bold")
    axes[2].set_ylabel("F1")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    root    = Path(__file__).parent
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)

    action_names = load_action_names(root / "Sample_Code")
    class_names  = [action_names[a] for a in SUBSET]
    n_classes    = len(SUBSET)

    # ── 1. Load and inspect natural distribution ──────────────────────────────
    all_samples = load_sensor_samples(root / "Inertial", SUBSET, LABEL_MAP)
    nat_counts  = Counter(y for _, y, _ in all_samples)

    print(f"Loaded {len(all_samples)} samples | {len(set(s for s,_,_ in all_samples))} subjects")
    print("\nNatural class distribution:")
    for i, name in enumerate(class_names):
        tag = "  ← will become minority" if i in MINORITY_IDX else ""
        print(f"  [{i}] {name:<28} {nat_counts[i]:>3} samples{tag}")
    print(f"\nImbalance ratio (natural): {max(nat_counts.values())/min(nat_counts.values()):.1f}:1  → balanced")

    # ── 2. Introduce imbalance ────────────────────────────────────────────────
    imb_samples = introduce_imbalance(all_samples, MINORITY_ACTS, KEEP_FRAC)
    imb_counts  = Counter(y for _, y, _ in imb_samples)

    print(f"\nAfter imbalance: {len(imb_samples)} / {len(all_samples)} samples remain")
    print("Imbalanced class distribution:")
    for i, name in enumerate(class_names):
        dropped = nat_counts[i] - imb_counts.get(i, 0)
        tag = f"  (dropped {dropped})" if dropped else ""
        print(f"  [{i}] {name:<28} {imb_counts.get(i,0):>3} samples{tag}")
    ratio = max(imb_counts.values()) / max(1, min(imb_counts.values()))
    print(f"\nImbalance ratio after: {ratio:.1f}:1")

    plot_distributions(all_samples, imb_samples, class_names,
                       out_dir / "part5_distributions.png")
    print("Saved part5_distributions.png")

    # ── 3. Train four variants with LOSO ─────────────────────────────────────
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else "cpu"
    )
    print(f"\nDevice: {device}  |  Epochs/fold: {N_EPOCHS}  |  Minority augment copies: {N_AUG}")

    variant_data = {
        "balanced":  all_samples,    # reference ceiling: full balanced data
        "baseline":  imb_samples,    # shows the problem: imbalanced, no fix
        "weighted":  imb_samples,    # Fix 1: inverse-frequency CE weights
        "augmented": imb_samples,    # Fix 2: targeted time-warp + amplitude-warp
    }

    results = {}
    for variant, data in variant_data.items():
        n_train_total = len(data) - len(data) // len(set(s for s,_,_ in data))
        print(f"\n{'─'*60}\nVariant: {variant}  (~{n_train_total} training samples/fold)")
        y_true, y_pred = run_loso(data, n_classes, variant, device)
        results[variant] = (y_true, y_pred)

        acc  = accuracy_score(y_true, y_pred)
        f1m  = f1_score(y_true, y_pred, average="macro")
        f1pc = f1_score(y_true, y_pred, average=None)
        min_f1 = f1pc[list(MINORITY_IDX)].mean()
        print(f"  Accuracy: {acc:.3f}  |  Macro F1: {f1m:.3f}  |  Minority F1: {min_f1:.3f}")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

        save_confusion_matrix(
            y_true, y_pred, class_names,
            title=f"Part5 [{variant}]  Acc={acc:.3f}  Macro-F1={f1m:.3f}",
            path=out_dir / f"part5_{variant}_confusion.png",
        )

    # ── 4. Summary ────────────────────────────────────────────────────────────
    plot_summary(results, class_names, out_dir / "part5_summary.png")
    print(f"\nSaved part5_summary.png")

    print(f"\n{'='*68}")
    print(f"{'Variant':<14} {'Accuracy':>9} {'Macro F1':>10} {'Minority F1':>13} {'vs baseline':>12}")
    print("─" * 68)
    baseline_f1 = f1_score(*results["baseline"], average="macro")
    for v in ["balanced", "baseline", "weighted", "augmented"]:
        y_true, y_pred = results[v]
        acc   = accuracy_score(y_true, y_pred)
        f1m   = f1_score(y_true, y_pred, average="macro")
        f1pc  = f1_score(y_true, y_pred, average=None)
        minf1 = f1pc[list(MINORITY_IDX)].mean()
        delta = f"+{f1m - baseline_f1:.3f}" if v != "baseline" else "—"
        print(f"  {v:<12} {acc:>9.3f} {f1m:>10.3f} {minf1:>13.3f} {delta:>12}")
    print("─" * 68)
    print("Minority F1 = mean F1 over the 3 artificially rare action classes")


if __name__ == "__main__":
    main()
