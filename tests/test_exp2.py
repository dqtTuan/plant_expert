import numpy as np
import pandas as pd
import pytest

from experiments.exp2_active import evaluate
from experiments.simulate_answers import simulate


MINI_KG = {
    "diseases": [{"id": f"c{i}"} for i in range(3)],
    "context_symptoms": [
        {"id": "s0", "question_vi": "q0", "p": {"c0": 0.98, "c1": 0.02, "c2": 0.5}},
        {"id": "s1", "question_vi": "q1", "p": {"c0": 0.5, "c1": 0.5, "c2": 0.5}},
        {"id": "s2", "question_vi": "q2", "p": {"c0": 0.95, "c1": 0.05, "c2": 0.5}},
    ],
}
CLASSES = ["c0", "c1", "c2"]


def make_synthetic_probs(n=40):
    rows = []
    for i in range(n):
        true_c = "c0" if i % 2 == 0 else "c1"
        # vision alone predicts the wrong class slightly (0.46 vs 0.52) with high entropy
        p0, p1 = (0.46, 0.52) if true_c == "c0" else (0.52, 0.46)
        rows.append({"image_id": f"img_{i}", "split": "blur", "label": true_c,
                     "p_c0": p0, "p_c1": p1, "p_c2": 0.02, "H": 0.80, "MI": 0.1})
    return pd.DataFrame(rows)


def test_simulate_deterministic_when_no_noise():
    # With p in {0.02, 0.98}, perturb=0, noise=0, over 200 images of c0 s0 is almost always "yes"
    labels = {f"im_{i}": "c0" for i in range(200)}
    df = simulate(labels, MINI_KG, classes=CLASSES, perturb=0.0, noise=0.0, seed=0)
    s0_yes = (df[df["symptom_id"] == "s0"]["answer"] == "yes").mean()
    assert s0_yes > 0.92


def test_simulate_noise_rate():
    # Extreme likelihoods (1.0 for c0 on s_det) so any non-'yes' comes from noise
    kg_det = {
        "diseases": [{"id": "c0"}],
        "context_symptoms": [{"id": "s_det", "question_vi": "q", "p": {"c0": 1.0}}],
    }
    # bypass clip by monkeypatching or checking noise corruption on p=0.98:
    # Expected raw 'no' ~0.02*(1-0.2)=0.016, plus noise 0.2*(0.5 flip + 0.5 unknown) = 0.20 -> non-yes ~ 0.216
    labels = {f"im_{i}": "c0" for i in range(2000)}
    df = simulate(labels, MINI_KG, classes=CLASSES, perturb=0.0, noise=0.2, seed=1)
    sub = df[df["symptom_id"] == "s0"]["answer"]
    unk_rate = (sub == "unknown").mean()
    # Half of noise (0.10) becomes "unknown"
    assert abs(unk_rate - 0.10) < 0.03


def test_ig_beats_vision_on_synthetic():
    probs_df = make_synthetic_probs(40)
    labels = dict(zip(probs_df["image_id"], probs_df["label"]))
    ans_df = simulate(labels, MINI_KG, classes=CLASSES, perturb=0.0, noise=0.0, seed=0)
    L = np.array([[s["p"][c] for c in CLASSES] for s in MINI_KG["context_symptoms"]])
    sym_ids = ["s0", "s1", "s2"]
    res_v = evaluate(probs_df, ans_df, L, sym_ids, CLASSES, tau=0.5, strategy="vision")
    res_ig = evaluate(probs_df, ans_df, L, sym_ids, CLASSES, tau=0.5, strategy="ig")
    assert res_ig["acc"] > res_v["acc"] + 0.5


def test_true_class_outside_candidates_counts_wrong():
    probs_df = pd.DataFrame([{
        "image_id": "hard1", "split": "blur", "label": "c2",
        "p_c0": 0.50, "p_c1": 0.49, "p_c2": 0.01, "H": 0.80, "MI": 0.1,
    }])
    ans_df = pd.DataFrame([
        {"image_id": "hard1", "symptom_id": "s0", "answer": "yes"},
        {"image_id": "hard1", "symptom_id": "s1", "answer": "yes"},
        {"image_id": "hard1", "symptom_id": "s2", "answer": "yes"},
    ])
    L = np.array([[s["p"][c] for c in CLASSES] for s in MINI_KG["context_symptoms"]])
    res = evaluate(probs_df, ans_df, L, ["s0", "s1", "s2"], CLASSES, tau=0.5, strategy="ig")
    assert res["acc"] == 0.0


def test_missing_answers_are_unknown():
    probs_df = make_synthetic_probs(10)
    empty_ans = pd.DataFrame(columns=["image_id", "symptom_id", "answer"])
    L = np.array([[s["p"][c] for c in CLASSES] for s in MINI_KG["context_symptoms"]])
    res_v = evaluate(probs_df, empty_ans, L, ["s0", "s1", "s2"], CLASSES, tau=0.5, strategy="vision")
    res_ig = evaluate(probs_df, empty_ans, L, ["s0", "s1", "s2"], CLASSES, tau=0.5, strategy="ig")
    assert res_ig["acc"] == pytest.approx(res_v["acc"])
    assert res_ig["avg_q"] <= 3.0
