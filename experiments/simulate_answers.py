"""Simulate farmer answers to context symptom questions from perturbed P(s|c) + answer noise."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plant_expert.kg import CLASSES, likelihood_matrix, load_kg


def simulate(
    labels: dict[str, str],
    kg: dict,
    classes: list[str] | None = None,
    perturb: float = 0.15,
    noise: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    cls_list = classes or CLASSES
    sym_ids, L = likelihood_matrix(kg, cls_list)
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-perturb, perturb, size=L.shape) if perturb > 0 else 0.0
    L_true = np.clip(L + delta, 0.02, 0.98)
    c_idx = {c: i for i, c in enumerate(cls_list)}
    rows = []
    for img_id, label in labels.items():
        ci = c_idx[label]
        for si, sid in enumerate(sym_ids):
            ans = "yes" if rng.random() < L_true[si, ci] else "no"
            if noise > 0 and rng.random() < noise:
                ans = "unknown" if rng.random() < 0.5 else ("no" if ans == "yes" else "yes")
            rows.append({"image_id": img_id, "symptom_id": sid, "answer": ans})
    return pd.DataFrame(rows, columns=["image_id", "symptom_id", "answer"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="artifacts/test_mc_probs.csv")
    ap.add_argument("--kg", default="kg/kg.json")
    ap.add_argument("--out", default="experiments/answers_sim.csv")
    ap.add_argument("--perturb", type=float, default=0.15)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = pd.read_csv(args.probs)
    kg = load_kg(args.kg)
    labels = dict(zip(df["image_id"], df["label"]))
    out_df = simulate(labels, kg, perturb=args.perturb, noise=args.noise, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved {len(out_df)} simulated answers to {args.out}")


if __name__ == "__main__":
    main()
