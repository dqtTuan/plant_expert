"""Active Bayesian questioning: pick the most informative context symptom, update the posterior."""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

EPS = 1e-12


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    return float(-(p * np.log(p + EPS)).sum())


def candidate_prior(p: np.ndarray, k: int = 3, min_p: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(-p)[:k]
    if len(idx) == 3 and p[idx[2]] < min_p:
        idx = idx[:2]
    return idx, p[idx] / p[idx].sum()


def update(prior: np.ndarray, lik: np.ndarray, answer: str) -> np.ndarray:
    if answer == "unknown":
        return prior.copy()
    l = lik if answer == "yes" else 1 - lik
    post = prior * l
    return post / post.sum() if post.sum() > 0 else prior.copy()


def info_gain(prior: np.ndarray, lik: np.ndarray) -> float:
    p_yes = float(prior @ lik)
    exp_h = sum(po * entropy(update(prior, lik, o)) for o, po in (("yes", p_yes), ("no", 1 - p_yes)) if po > EPS)
    return entropy(prior) - exp_h


@dataclass
class State:
    cand: np.ndarray
    post: np.ndarray
    asked: list = field(default_factory=list)
    history: list = field(default_factory=list)  # (symptom idx, answer, posterior after)
    gated: bool = False


def start(p: np.ndarray, H: float, tau: float, k: int = 3) -> State:
    p = np.asarray(p, dtype=float)
    if H <= tau:
        return State(cand=np.arange(len(p)), post=p.copy(), gated=False)
    cand, prior = candidate_prior(p, k)
    return State(cand=cand, post=prior, gated=True)


def is_done(st: State, conf: float = 0.85, max_q: int = 3) -> bool:
    return (not st.gated) or st.post.max() >= conf or len(st.asked) >= max_q


def next_question(st: State, L: np.ndarray, strategy: str = "ig", rng: np.random.Generator | None = None) -> int | None:
    if is_done(st):
        return None
    free = [s for s in range(L.shape[0]) if s not in st.asked]
    if not free:
        return None
    if strategy == "random":
        return int((rng or np.random.default_rng()).choice(free))
    gains = [info_gain(st.post, L[s, st.cand]) for s in free]
    best = int(np.argmax(gains))
    return free[best] if gains[best] > 1e-9 else None


def answer(st: State, s: int, ans: str, L: np.ndarray) -> State:
    post = update(st.post, L[s, st.cand], ans)
    return State(cand=st.cand, post=post, asked=st.asked + [s],
                 history=st.history + [(s, ans, post)], gated=st.gated)


def final(st: State) -> tuple[int, float, bool]:
    i = int(np.argmax(st.post))
    prob = float(st.post[i])
    return int(st.cand[i]), prob, bool((not st.gated) or prob >= 0.85)


def run_dialogue(p, H, tau, L, answer_fn: Callable[[int], str], strategy: str = "ig",
                 rng: np.random.Generator | None = None) -> tuple[int, int]:
    st = start(p, H, tau)
    while (s := next_question(st, L, strategy, rng)) is not None:
        st = answer(st, s, answer_fn(s), L)
    return final(st)[0], len(st.asked)
