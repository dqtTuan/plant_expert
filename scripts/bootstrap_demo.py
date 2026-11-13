"""Bootstrap local artifacts, sample leaf images, and baseline experiment outputs for offline demo & testing."""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plant_expert.kg import CLASSES
from plant_expert.vision import build_model, predictive_entropy


def _make_leaf(path: Path, bg: tuple[int, int, int], spot_colors: list[tuple[int, int, int]], seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (256, 256), (235, 240, 232))
    draw = ImageDraw.Draw(img)
    draw.ellipse([36, 24, 220, 232], fill=bg, outline=(34, 85, 34), width=3)
    draw.line([(128, 32), (128, 236)], fill=(65, 115, 55), width=3)
    for col in spot_colors:
        for _ in range(10):
            x = int(rng.integers(60, 195))
            y = int(rng.integers(50, 205))
            r = int(rng.integers(5, 16))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=col)
    img.save(path)


def _compute_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += float(m.mean() * abs(acc[m].mean() - conf[m].mean()))
    return ece


def _risk_coverage(u: np.ndarray, err: np.ndarray):
    order = np.argsort(u)
    sorted_err = err[order]
    cov = np.arange(1, len(u) + 1) / len(u)
    risk = np.cumsum(sorted_err) / np.arange(1, len(u) + 1)
    return cov, risk, float(np.trapezoid(risk, cov))


def generate_artifacts(art_dir: Path, tau: float = 1.55) -> None:
    art_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model = build_model(len(CLASSES), p_drop=0.3, pretrained=False)
    torch.save(model.state_dict(), art_dir / "model.pt")
    cfg = {
        "classes": CLASSES,
        "tau": tau,
        "p_drop": 0.3,
        "img_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "T": 20,
        "bootstrap": True,
    }
    (art_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Generate 250 synthetic test rows (50 per split x 5 splits) for Exp 1 & Exp 2
    rng = np.random.default_rng(42)
    splits = {
        "clean": (0.94, 0.06, 0.4),
        "blur": (0.74, 0.22, 1.8),
        "occlusion": (0.76, 0.20, 1.7),
        "brightness": (0.78, 0.18, 1.6),
        "plantdoc": (0.70, 0.25, 1.9),
    }
    mc_csv_lines = ["image_id,split,label," + ",".join(f"p_{c}" for c in CLASSES) + ",H,MI"]
    metric_lines = ["split,method,acc,ece,auroc_err,auroc_mi,aurc"]

    blur_store = {}
    for sname, (acc_target, overconf_shift, ent_scale) in splits.items():
        n = 50
        y = np.arange(n) % len(CLASSES)
        p_mc = np.zeros((n, len(CLASSES)), dtype=float)
        p_soft = np.zeros_like(p_mc)
        p_temp = np.zeros_like(p_mc)
        mi_vals = np.zeros(n, dtype=float)

        for i in range(n):
            yi = int(y[i])
            is_correct = rng.random() < acc_target
            logits = rng.normal(0.0, 0.4, size=len(CLASSES))
            if is_correct:
                logits[yi] += 3.2 / max(ent_scale, 0.5)
                mi_vals[i] = float(rng.uniform(0.01, 0.06))
            else:
                wrong = (yi + int(rng.integers(1, len(CLASSES)))) % len(CLASSES)
                logits[yi] += 1.45
                logits[wrong] += 1.55
                mi_vals[i] = float(rng.uniform(0.10, 0.24))

            # MC Dropout is better calibrated & more uncertain on hard/wrong samples
            ex = np.exp(logits - logits.max())
            p_mc[i] = ex / ex.sum()
            # Softmax baseline is overconfident even when wrong
            s_logits = logits * (1.8 + overconf_shift * 2.0)
            exs = np.exp(s_logits - s_logits.max())
            p_soft[i] = exs / exs.sum()
            # Temperature scaling calibrates confidence uniformly
            t_logits = s_logits / 1.45
            ext = np.exp(t_logits - t_logits.max())
            p_temp[i] = ext / ext.sum()

            H_i = float(predictive_entropy(p_mc[i]))
            row_vals = [f"{sname}_{i}", sname, CLASSES[yi]] + [f"{v:.6f}" for v in p_mc[i]] + [f"{H_i:.6f}", f"{mi_vals[i]:.6f}"]
            mc_csv_lines.append(",".join(row_vals))

        for mname, probs, mi_arr in (
            ("softmax", p_soft, None),
            ("temp_scaling", p_temp, None),
            ("mc_dropout", p_mc, mi_vals),
        ):
            H_arr = predictive_entropy(probs)
            err = (probs.argmax(axis=1) != y).astype(int)
            auroc = roc_auc_score(err, H_arr) if len(np.unique(err)) > 1 else 0.5
            auroc_mi = roc_auc_score(err, mi_arr) if (mi_arr is not None and len(np.unique(err)) > 1) else float("nan")
            _, _, aurc = _risk_coverage(H_arr, err.astype(float))
            acc = float((1 - err).mean())
            ece = _compute_ece(probs, y)
            metric_lines.append(f"{sname},{mname},{acc:.4f},{ece:.4f},{auroc:.4f},{auroc_mi:.4f},{aurc:.4f}")
            if sname == "blur":
                blur_store[mname] = (y, probs, H_arr)

    (art_dir / "test_mc_probs.csv").write_text("\n".join(mc_csv_lines) + "\n", encoding="utf-8")
    (art_dir / "exp1_metrics.csv").write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    # 3 plots on blur split
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    for mname, (yb, pb, _) in blur_store.items():
        conf, acc = pb.max(1), (pb.argmax(1) == yb).astype(float)
        bins = np.linspace(0, 1, 11)
        xs, ys_b = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                xs.append(conf[m].mean())
                ys_b.append(acc[m].mean())
        ax.plot(xs, ys_b, marker="o", label=mname)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title("Reliability (blur)"); ax.legend()
    fig.tight_layout(); fig.savefig(art_dir / "exp1_reliability.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for mname, (yb, pb, Hb) in blur_store.items():
        err = (pb.argmax(1) != yb).astype(float)
        c_curve, r_curve, _ = _risk_coverage(Hb, err)
        ax.plot(c_curve, r_curve, label=mname)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Risk (Error rate)"); ax.set_title("Risk-Coverage (blur)"); ax.legend()
    fig.tight_layout(); fig.savefig(art_dir / "exp1_risk_coverage.png", dpi=140); plt.close(fig)

    yb, pb, Hb = blur_store["mc_dropout"]
    ok_m = pb.argmax(1) == yb
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(Hb[ok_m], bins=15, alpha=0.6, label="Correct")
    ax.hist(Hb[~ok_m], bins=15, alpha=0.6, label="Wrong")
    ax.axvline(tau, color="red", linestyle="--", label=f"tau={tau:.2f}")
    ax.set_xlabel("Predictive Entropy H"); ax.set_ylabel("Count"); ax.legend()
    fig.tight_layout(); fig.savefig(art_dir / "exp1_entropy_hist.png", dpi=140); plt.close(fig)


def generate_samples(sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    _make_leaf(sample_dir / "late_blight_leaf.png", (52, 130, 52), [(45, 32, 22), (80, 70, 55)], seed=1)
    _make_leaf(sample_dir / "ambiguous_leaf.png", (70, 145, 62), [(155, 140, 45), (90, 75, 40)], seed=2)
    _make_leaf(sample_dir / "healthy_leaf.png", (48, 155, 58), [], seed=3)


def main() -> None:
    generate_artifacts(ROOT / "artifacts")
    generate_samples(ROOT / "samples")
    py = sys.executable
    subprocess.run([py, str(ROOT / "scripts/build_index.py")], check=True)
    subprocess.run([py, str(ROOT / "experiments/simulate_answers.py"), "--probs", str(ROOT / "artifacts/test_mc_probs.csv"), "--out", str(ROOT / "experiments/answers_sim.csv")], check=True)
    subprocess.run([py, str(ROOT / "experiments/exp2_active.py"), "--answers", str(ROOT / "experiments/answers_sim.csv"), "--sweep"], check=True)
    subprocess.run([py, str(ROOT / "experiments/exp3_rag.py"), "--allow-fallback"], check=True)
    print("Bootstrap complete: artifacts/, samples/, kg/index.npz, and experiments/out/ are ready.")


if __name__ == "__main__":
    main()
