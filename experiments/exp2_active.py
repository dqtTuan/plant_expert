"""Experiment 2: evaluate Vision-only vs Random Questioning vs Active Bayesian Questioning (IG)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plant_expert  # noqa: E402, F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from plant_expert.bayes import run_dialogue
from plant_expert.kg import CLASSES, likelihood_matrix, load_kg
from experiments.simulate_answers import simulate


def evaluate(
    probs_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    L: np.ndarray,
    sym_ids: list[str],
    classes: list[str],
    tau: float,
    strategy: str = "ig",
) -> dict:
    hard = probs_df[probs_df["H"] > tau].copy()
    if len(hard) == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "avg_q": 0.0, "n": 0}
    ans_map = {
        (r["image_id"], r["symptom_id"]): r["answer"]
        for _, r in answers_df.iterrows()
    }
    c_idx = {c: i for i, c in enumerate(classes)}
    p_cols = [f"p_{c}" for c in classes]
    rng = np.random.default_rng(0)
    y_true, y_pred, q_counts = [], [], []
    for _, row in hard.iterrows():
        p = np.array([row[col] for col in p_cols], dtype=float)
        y_true.append(c_idx[row["label"]])
        if strategy == "vision":
            y_pred.append(int(np.argmax(p)))
            q_counts.append(0)
        else:
            img_id = row["image_id"]
            pred, n_q = run_dialogue(
                p, float(row["H"]), tau, L,
                answer_fn=lambda s, iid=img_id: ans_map.get((iid, sym_ids[s]), "unknown"),
                strategy=strategy,
                rng=rng,
            )
            y_pred.append(pred)
            q_counts.append(n_q)
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    return {
        "acc": float((y_t == y_p).mean()),
        "macro_f1": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        "avg_q": float(np.mean(q_counts)),
        "n": int(len(hard)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="artifacts/test_mc_probs.csv")
    ap.add_argument("--config", default="artifacts/config.json")
    ap.add_argument("--kg", default="kg/kg.json")
    ap.add_argument("--answers", default="experiments/answers_sim.csv")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    probs_df = pd.read_csv(args.probs)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tau = float(cfg["tau"])
    classes = cfg.get("classes", CLASSES)
    kg = load_kg(args.kg)
    sym_ids, L = likelihood_matrix(kg, classes)
    ans_df = pd.read_csv(args.answers)

    is_sim = "sim" in Path(args.answers).stem.lower()
    suffix = "_simulated" if is_sim else ""
    out_dir = Path("experiments/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for strat in ("vision", "random", "ig"):
        m = evaluate(probs_df, ans_df, L, sym_ids, classes, tau, strategy=strat)
        rows.append({"strategy": strat, "source": "simulated" if is_sim else "real", **m})
    res_df = pd.DataFrame(rows)
    res_path = out_dir / f"exp2_results{suffix}.csv"
    res_df.to_csv(res_path, index=False)
    print(res_df.to_string(index=False))

    if args.sweep:
        labels = dict(zip(probs_df["image_id"], probs_df["label"]))
        sweep_rows = []
        for pert in (0.0, 0.15, 0.30):
            for ns in (0.0, 0.10, 0.20, 0.30):
                sim_df = simulate(labels, kg, classes=classes, perturb=pert, noise=ns, seed=0)
                for strat in ("vision", "random", "ig"):
                    m = evaluate(probs_df, sim_df, L, sym_ids, classes, tau, strategy=strat)
                    sweep_rows.append({"perturb": pert, "noise": ns, "strategy": strat, **m})
        sw_df = pd.DataFrame(sweep_rows)
        sw_df.to_csv(out_dir / "exp2_sweep_simulated.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        for pert in (0.0, 0.15, 0.30):
            sub = sw_df[(sw_df["perturb"] == pert) & (sw_df["strategy"] == "ig")]
            ax.plot(sub["noise"], sub["acc"], marker="o", label=f"IG (perturb={pert})")
        v_acc = sw_df[sw_df["strategy"] == "vision"]["acc"].iloc[0]
        ax.axhline(v_acc, linestyle="--", color="gray", label="Vision-only")
        ax.set_xlabel("Answer noise rate"); ax.set_ylabel("Hard-subset Top-1 Accuracy")
        ax.set_title("Exp 2 Sensitivity (Simulated Answers)"); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "exp2_sweep_simulated.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
