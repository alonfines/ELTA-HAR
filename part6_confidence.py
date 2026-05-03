"""
Part 6 — Confidence Estimation with Statistical Guarantees

Two post-hoc techniques applied to retrained sensor and video transformers:

  1. Temperature Scaling — learn a scalar T that divides logits before softmax.
     Reduces ECE without changing accuracy.  Fit by minimising NLL on the
     held-out calibration subject.

  2. Split Conformal Prediction — distribution-free prediction sets with
     finite-sample coverage guarantee  P(y ∈ C(x)) ≥ 1−α.
     Non-conformity score:  s_i = 1 − softmax(z_i / T*)[y_i]
     Calibration quantile:  q̂ = sort(s_cal)[ ceil((n+1)(1−α)) − 1 ]
     Prediction set:        C(x) = { y : softmax(z/T*)[y] ≥ 1 − q̂ }

Data split — 3-way cyclic LOSO (8 folds):
  fold i:  test = s_i,  cal = s_{i+1 mod 8},  train = remaining 6

Run: python part6_confidence.py
"""

import random
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from data.sensor_dataset import SensorDataset, load_sensor_samples, pad_or_truncate
from data.video_dataset import PoseDataset, load_video_samples
from models.fusion import FusionMLP
from train import SensorLightningModule, VideoLightningModule
from utils import load_action_names, load_config, save_confusion_matrix
from utils_ml import (
    fit_temperature, compute_ece, nonconf_scores, cp_quantile, cp_predict_sets,
    empirical_coverage, mean_set_size
)

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SCFG = load_config(str(ROOT / "configs/sensor.yaml"))
VCFG = load_config(str(ROOT / "configs/video.yaml"))

SUBSET     = SCFG.subset
LABEL_MAP  = {a: i for i, a in enumerate(SUBSET)}
KEEP_FEATS = [i for i in range(66) if i not in VCFG.drop_feats]
ALPHAS     = [0.05, 0.10, 0.20]
SEED       = 42

F_P_DROP   = 0.3
F_N_EPOCHS = 100
F_LR       = 1e-3
F_WD       = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else "cpu"
)
print(f"Device: {DEVICE}")


# ── Inference helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def get_logits(
    module: pl.LightningModule,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw pre-softmax logits (N, C) and integer labels (N,)."""
    module.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        all_logits.append(module(x.to(device)).cpu().numpy())
        all_y.extend(y.numpy())
    return np.concatenate(all_logits, axis=0), np.array(all_y)


# ── Temperature Scaling ────────────────────────────────────────────────────────

# All utilities imported from utils_ml (see imports above)


# ── Training helper ────────────────────────────────────────────────────────────

def train_module(module: pl.LightningModule, loader: DataLoader, n_epochs: int) -> None:
    pl.Trainer(
        max_epochs=n_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="auto",
        gradient_clip_val=SCFG.training.grad_clip,
    ).fit(module, loader)


# ── Fusion helpers ─────────────────────────────────────────────────────────────

@torch.no_grad()
def get_fusion_embeddings(s_mod, v_mod, s_samples, v_samples, max_len, device):
    """Extract paired (N, d_emb) encoder embeddings for both modalities."""
    s_mod.eval(); v_mod.eval()
    e_v_list, e_s_list, y_list = [], [], []
    for (sy, raw), (_, seq) in zip(s_samples, v_samples):
        # sensor: pad/truncate → add velocity → z-score → encode
        x_s = pad_or_truncate(raw, max_len).astype(np.float32)
        vel = np.zeros_like(x_s); vel[1:] = x_s[1:] - x_s[:-1]
        x_s = np.concatenate([x_s, vel], axis=-1)
        mu, std = x_s.mean(0, keepdims=True), x_s.std(0, keepdims=True) + 1e-8
        x_s = torch.from_numpy((x_s - mu) / std).unsqueeze(0).to(device)
        sm  = s_mod.model
        e_s = sm.encoder(sm.pos_enc(sm.proj(x_s))).mean(1).squeeze(0).cpu().numpy()

        # video: add velocity → z-score → encode
        x_v = seq.astype(np.float32)
        vel = np.zeros_like(x_v); vel[1:] = x_v[1:] - x_v[:-1]
        x_v = np.concatenate([x_v, vel], axis=-1)
        mu, std = x_v.mean(0, keepdims=True), x_v.std(0, keepdims=True) + 1e-8
        x_v = torch.from_numpy((x_v - mu) / std).unsqueeze(0).to(device)
        vm  = v_mod.model
        e_v = vm.encoder(vm.pos_enc(vm.proj(x_v))).mean(1).squeeze(0).cpu().numpy()

        e_s_list.append(e_s); e_v_list.append(e_v); y_list.append(sy)
    return np.stack(e_v_list), np.stack(e_s_list), np.array(y_list)


def train_fusion_head(e_v, e_s, y, n_classes, device):
    """Train FusionMLP on pre-extracted embeddings with modality dropout."""
    ev = torch.tensor(e_v, dtype=torch.float32).to(device)
    es = torch.tensor(e_s, dtype=torch.float32).to(device)
    yt = torch.tensor(y,   dtype=torch.long).to(device)
    model  = FusionMLP(n_classes).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=F_LR, weight_decay=F_WD)
    loader = DataLoader(TensorDataset(ev, es, yt), batch_size=32, shuffle=True)
    for _ in range(F_N_EPOCHS):
        model.train()
        for bv, bs, by in loader:
            B  = bv.size(0)
            dv = (torch.rand(B) < F_P_DROP).to(device)
            ds = (torch.rand(B) < F_P_DROP).to(device)
            ds[dv & ds] = False          # ensure at least one modality per sample
            loss = F.cross_entropy(model(bv, bs, dv, ds), by)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


@torch.no_grad()
def get_fusion_logits(fusion_model, e_v, e_s, device):
    ev = torch.tensor(e_v, dtype=torch.float32).to(device)
    es = torch.tensor(e_s, dtype=torch.float32).to(device)
    return fusion_model(ev, es).cpu().numpy()


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_reliability(
    agg: dict,   # {"Sensor": {"before": (N,C), "after": (N,C), "y": (N,)}, "Video": ...}
    path: Path,
) -> None:
    n_bins = 10
    edges  = np.linspace(0, 1, n_bins + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Reliability Diagrams — Before vs After Temperature Scaling",
                 fontweight="bold")

    for ax, (modality, data) in zip(axes, agg.items()):
        y = data["y"]
        for label, logits, color, ls in [
            ("Before (raw)",  data["before"], "steelblue",  "--"),
            ("After (TS)",    data["after"],  "darkorange", "-"),
        ]:
            probs   = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
            conf    = probs.max(axis=1)
            correct = (probs.argmax(axis=1) == y).astype(float)
            bin_conf, bin_acc = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (conf > lo) & (conf <= hi)
                if mask.sum() == 0:
                    continue
                bin_conf.append(conf[mask].mean())
                bin_acc.append(correct[mask].mean())
            ax.plot(bin_conf, bin_acc, marker="o", color=color, linestyle=ls,
                    label=label, linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Perfect")
        ax.set_title(f"{modality} model", fontweight="bold")
        ax.set_xlabel("Mean confidence"); ax.set_ylabel("Fraction correct")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def plot_ece(ece_folds: dict, path: Path) -> None:
    modalities = list(ece_folds.keys())
    n          = len(ece_folds[modalities[0]]["before"])
    x, w       = np.arange(n), 0.3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ECE per Fold — Before vs After Temperature Scaling",
                 fontweight="bold")

    for ax, mod in zip(axes, modalities):
        before = ece_folds[mod]["before"]
        after  = ece_folds[mod]["after"]
        ax.bar(x - w/2, before, w, label="Before (raw)", color="steelblue",  alpha=0.8)
        ax.bar(x + w/2, after,  w, label="After (TS)",  color="darkorange", alpha=0.8)
        ax.axhline(np.mean(before), color="steelblue",  linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axhline(np.mean(after),  color="darkorange", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{i+1}" for i in range(n)], fontsize=9)
        ax.set_title(
            f"{mod}  (mean before={np.mean(before):.3f}, after={np.mean(after):.3f})",
            fontweight="bold",
        )
        ax.set_ylabel("ECE"); ax.set_ylim(0, max(max(before), max(after)) * 1.35)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def plot_coverage(
    cp_summary: dict,   # {"Sensor": {"cov": {α: val}, "size": {α: val}}, ...}
    path: Path,
) -> None:
    colors = {"Sensor": "steelblue", "Video": "darkorange", "Fusion": "forestgreen"}
    x_vals = [1 - a for a in ALPHAS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Conformal Prediction — Coverage and Prediction-Set Size",
                 fontweight="bold")

    for ax, (metric, ylabel, title) in zip(axes, [
        ("cov",  "Empirical coverage",  "Coverage  (guarantee: ≥ 1−α)"),
        ("size", "Mean prediction-set size", "Set size  (efficiency)"),
    ]):
        for mod, data in cp_summary.items():
            vals = [data[metric][a] for a in ALPHAS]
            ax.plot(x_vals, vals, marker="o", color=colors[mod],
                    linewidth=2, label=mod)

        if metric == "cov":
            for x, a in zip(x_vals, ALPHAS):
                ax.axhline(1 - a, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylim(0.55, 1.05)

        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"1−{a}\n(α={a})" for a in ALPHAS], fontsize=9)
        ax.set_xlabel("Target coverage (1−α)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    action_names = load_action_names(ROOT / "Sample_Code")
    class_names  = [action_names[a] for a in SUBSET]
    n_classes    = len(SUBSET)
    in_dim_s     = 12                    # 6 IMU + 6 velocity
    in_dim_v     = len(KEEP_FEATS) * 2  # pose position + velocity

    print("Loading data...")
    sensor_samples = load_sensor_samples(ROOT / SCFG.inertial_dir, SUBSET, LABEL_MAP)
    video_samples  = load_video_samples(ROOT / VCFG.kalman_cache, SUBSET, LABEL_MAP, KEEP_FEATS)
    subjects       = sorted(set(s for s, _, _ in sensor_samples))
    print(f"  {len(sensor_samples)} sensor | {len(video_samples)} video | {len(subjects)} subjects\n")

    bs = SCFG.training.batch_size

    # ── Accumulators ───────────────────────────────────────────────────────────
    ece_folds = {mod: {"before": [], "after": [], "T": []} for mod in ("Sensor", "Video")}
    cp_folds  = {mod: {a: {"cov": [], "size": []} for a in ALPHAS}
                 for mod in ("Sensor", "Video", "Fusion")}
    agg_logits = {mod: {"before": [], "after": [], "y": []}
                  for mod in ("Sensor", "Video")}
    agg_pred   = {mod: {"y_true": [], "y_pred": []} for mod in ("Sensor", "Video", "Fusion")}

    for i, test_subj in enumerate(subjects):
        cal_subj    = subjects[(i + 1) % len(subjects)]
        train_subjs = set(subjects) - {test_subj, cal_subj}
        print(f"── Fold {i+1}/8  test=s{test_subj}  cal=s{cal_subj}  "
              f"train={{s{',s'.join(str(s) for s in sorted(train_subjs))}}}")

        # ── Split ──────────────────────────────────────────────────────────────
        s_tr, s_ca, s_te = (
            [(y, r) for s, y, r in sensor_samples if s in train_subjs],
            [(y, r) for s, y, r in sensor_samples if s == cal_subj],
            [(y, r) for s, y, r in sensor_samples if s == test_subj],
        )
        v_tr, v_ca, v_te = (
            [(y, seq) for s, y, seq in video_samples if s in train_subjs],
            [(y, seq) for s, y, seq in video_samples if s == cal_subj],
            [(y, seq) for s, y, seq in video_samples if s == test_subj],
        )

        def make_loaders(tr, ca, te, Cls, kwargs):
            return (
                DataLoader(Cls(tr, augment=True,  **kwargs), batch_size=bs, shuffle=True,  num_workers=0),
                DataLoader(Cls(ca, augment=False, **kwargs), batch_size=bs, shuffle=False, num_workers=0),
                DataLoader(Cls(te, augment=False, **kwargs), batch_size=bs, shuffle=False, num_workers=0),
            )

        s_loaders = make_loaders(s_tr, s_ca, s_te, SensorDataset, {"max_len": SCFG.max_len})
        v_loaders = make_loaders(v_tr, v_ca, v_te, PoseDataset,   {})

        # ── Train ──────────────────────────────────────────────────────────────
        torch.manual_seed(SEED)
        s_mod = SensorLightningModule(
            n_classes=n_classes, in_dim=in_dim_s,
            d_model=SCFG.model.d_model, n_heads=SCFG.model.n_heads,
            n_layers=SCFG.model.n_layers, dropout=SCFG.model.dropout,
            lr=SCFG.training.lr, weight_decay=SCFG.training.weight_decay,
            n_epochs=SCFG.training.n_epochs,
        )
        train_module(s_mod, s_loaders[0], SCFG.training.n_epochs)
        s_mod = s_mod.to(DEVICE)

        torch.manual_seed(SEED)
        v_mod = VideoLightningModule(
            n_classes=n_classes, in_dim=in_dim_v,
            d_model=VCFG.model.d_model, n_heads=VCFG.model.n_heads,
            n_layers=VCFG.model.n_layers, dropout=VCFG.model.dropout,
            lr=VCFG.training.lr, weight_decay=VCFG.training.weight_decay,
            n_epochs=VCFG.training.n_epochs,
        )
        train_module(v_mod, v_loaders[0], VCFG.training.n_epochs)
        v_mod = v_mod.to(DEVICE)

        # ── Logits ─────────────────────────────────────────────────────────────
        s_cal_logits, s_cal_y   = get_logits(s_mod, s_loaders[1], DEVICE)
        s_te_logits,  s_te_y    = get_logits(s_mod, s_loaders[2], DEVICE)
        v_cal_logits, v_cal_y   = get_logits(v_mod, v_loaders[1], DEVICE)
        v_te_logits,  v_te_y    = get_logits(v_mod, v_loaders[2], DEVICE)

        # ── Temperature Scaling ────────────────────────────────────────────────
        T_s = fit_temperature(s_cal_logits, s_cal_y)
        T_v = fit_temperature(v_cal_logits, v_cal_y)

        for mod, te_logits, te_y, T in [
            ("Sensor", s_te_logits, s_te_y, T_s),
            ("Video",  v_te_logits, v_te_y, T_v),
        ]:
            ece_folds[mod]["before"].append(compute_ece(te_logits, te_y, T=1.0))
            ece_folds[mod]["after"].append(compute_ece(te_logits,  te_y, T=T))
            ece_folds[mod]["T"].append(T)
            agg_logits[mod]["before"].append(te_logits)
            agg_logits[mod]["after"].append(te_logits / T)   # store scaled logits
            agg_logits[mod]["y"].extend(te_y)
            agg_pred[mod]["y_true"].extend(te_y)
            agg_pred[mod]["y_pred"].extend(te_logits.argmax(axis=1))

        # ── Conformal Prediction ───────────────────────────────────────────────
        for alpha in ALPHAS:
            for mod, cal_logits, cal_y, te_logits, te_y, T in [
                ("Sensor", s_cal_logits, s_cal_y, s_te_logits, s_te_y, T_s),
                ("Video",  v_cal_logits, v_cal_y, v_te_logits, v_te_y, T_v),
            ]:
                scores = nonconf_scores(cal_logits, cal_y, T)
                q_hat  = cp_quantile(scores, alpha)
                sets   = cp_predict_sets(te_logits, T, q_hat)
                cp_folds[mod][alpha]["cov"].append(empirical_coverage(sets, te_y))
                cp_folds[mod][alpha]["size"].append(mean_set_size(sets))

        s_acc = accuracy_score(s_te_y, s_te_logits.argmax(axis=1))
        v_acc = accuracy_score(v_te_y, v_te_logits.argmax(axis=1))
        print(f"  Sensor  T*={T_s:.3f}  ECE {ece_folds['Sensor']['before'][-1]:.3f}"
              f" → {ece_folds['Sensor']['after'][-1]:.3f}  acc={s_acc:.3f}")
        print(f"  Video   T*={T_v:.3f}  ECE {ece_folds['Video']['before'][-1]:.3f}"
              f" → {ece_folds['Video']['after'][-1]:.3f}  acc={v_acc:.3f}")

        # ── Fusion model ───────────────────────────────────────────────────────
        s_tr_p = [(y, r)   for s, y, r   in sensor_samples if s in train_subjs]
        s_ca_p = [(y, r)   for s, y, r   in sensor_samples if s == cal_subj]
        s_te_p = [(y, r)   for s, y, r   in sensor_samples if s == test_subj]
        v_tr_p = [(y, seq) for s, y, seq in video_samples  if s in train_subjs]
        v_ca_p = [(y, seq) for s, y, seq in video_samples  if s == cal_subj]
        v_te_p = [(y, seq) for s, y, seq in video_samples  if s == test_subj]

        e_v_tr, e_s_tr, y_ftr = get_fusion_embeddings(
            s_mod, v_mod, s_tr_p, v_tr_p, SCFG.max_len, DEVICE)
        e_v_ca, e_s_ca, y_fca = get_fusion_embeddings(
            s_mod, v_mod, s_ca_p, v_ca_p, SCFG.max_len, DEVICE)
        e_v_te, e_s_te, y_fte = get_fusion_embeddings(
            s_mod, v_mod, s_te_p, v_te_p, SCFG.max_len, DEVICE)

        torch.manual_seed(SEED)
        fusion_mod = train_fusion_head(e_v_tr, e_s_tr, y_ftr, n_classes, DEVICE)

        f_cal_logits = get_fusion_logits(fusion_mod, e_v_ca, e_s_ca, DEVICE)
        f_te_logits  = get_fusion_logits(fusion_mod, e_v_te, e_s_te, DEVICE)

        for alpha in ALPHAS:
            scores = nonconf_scores(f_cal_logits, y_fca, T=1.0)
            q_hat  = cp_quantile(scores, alpha)
            sets   = cp_predict_sets(f_te_logits, 1.0, q_hat)
            cp_folds["Fusion"][alpha]["cov"].append(empirical_coverage(sets, y_fte))
            cp_folds["Fusion"][alpha]["size"].append(mean_set_size(sets))

        f_acc = accuracy_score(y_fte, f_te_logits.argmax(axis=1))
        print(f"  Fusion  acc={f_acc:.3f}")
        agg_pred["Fusion"]["y_true"].extend(y_fte.tolist())
        agg_pred["Fusion"]["y_pred"].extend(f_te_logits.argmax(axis=1).tolist())

    # ── Aggregate ──────────────────────────────────────────────────────────────
    for mod in ("Sensor", "Video"):
        agg_logits[mod]["before"] = np.concatenate(agg_logits[mod]["before"])
        agg_logits[mod]["after"]  = np.concatenate(agg_logits[mod]["after"])
        agg_logits[mod]["y"]      = np.array(agg_logits[mod]["y"])

    cp_summary = {
        mod: {
            metric: {a: np.mean(cp_folds[mod][a][metric]) for a in ALPHAS}
            for metric in ("cov", "size")
        }
        for mod in ("Sensor", "Video", "Fusion")
    }

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*82}")
    print("Part 6 — Confidence Estimation with Statistical Guarantees")
    print(f"{'='*82}")
    header_cp = "  ".join(f"α={a}  cov / size" for a in ALPHAS)
    print(f"\n{'Model':<10} {'ECE_raw':>9} {'ECE_TS':>8} {'T*':>6}  {header_cp}")
    print("─" * 82)
    for mod in ("Sensor", "Video"):
        ece_b  = np.mean(ece_folds[mod]["before"])
        ece_a  = np.mean(ece_folds[mod]["after"])
        T_mean = np.mean(ece_folds[mod]["T"])
        cp_str = "  ".join(
            f"{cp_summary[mod]['cov'][a]:.3f} / {cp_summary[mod]['size'][a]:.2f}"
            for a in ALPHAS
        )
        print(f"  {mod:<8} {ece_b:>9.4f} {ece_a:>8.4f} {T_mean:>6.3f}  {cp_str}")
    cp_str_f = "  ".join(
        f"{cp_summary['Fusion']['cov'][a]:.3f} / {cp_summary['Fusion']['size'][a]:.2f}"
        for a in ALPHAS
    )
    print(f"  {'Fusion':<8} {'—':>9} {'—':>8} {'—':>6}  {cp_str_f}")
    print("─" * 82)
    print("Coverage guarantee: empirical coverage ≥ 1−α  (distribution-free, finite-sample)")
    print(f"  α=0.05 → need cov ≥ 0.950 | α=0.10 → need cov ≥ 0.900 | α=0.20 → need cov ≥ 0.800")

    # ── Plots ───────────────────────────────────────────────────────────────────
    plot_reliability(agg_logits,  OUT_DIR / "part6_reliability.png")
    plot_ece(
        {mod: {"before": ece_folds[mod]["before"], "after": ece_folds[mod]["after"]}
         for mod in ("Sensor", "Video")},
        OUT_DIR / "part6_ece.png",
    )
    plot_coverage(cp_summary, OUT_DIR / "part6_coverage.png")

    for mod in ("Sensor", "Video"):
        y_true = np.array(agg_pred[mod]["y_true"])
        y_pred = np.array(agg_pred[mod]["y_pred"])
        acc    = accuracy_score(y_true, y_pred)
        save_confusion_matrix(
            y_true, y_pred, class_names,
            title=f"Part 6 — {mod}  Acc={acc:.3f}",
            path=OUT_DIR / f"part6_{mod.lower()}_confusion.png",
            cmap="Blues" if mod == "Sensor" else "Greens",
        )

    y_true_f = np.array(agg_pred["Fusion"]["y_true"])
    y_pred_f = np.array(agg_pred["Fusion"]["y_pred"])
    f_acc    = accuracy_score(y_true_f, y_pred_f)
    save_confusion_matrix(
        y_true_f, y_pred_f, class_names,
        title=f"Part 6 — Fusion  Acc={f_acc:.3f}",
        path=OUT_DIR / "part6_fusion_confusion.png",
        cmap="Purples",
    )

    print("\nDone. Outputs saved to outputs/")


if __name__ == "__main__":
    main()
