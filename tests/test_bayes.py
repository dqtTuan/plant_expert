import numpy as np
import pytest

from plant_expert.bayes import (
    answer, candidate_prior, entropy, final, info_gain, is_done, next_question,
    run_dialogue, start, update,
)


def test_entropy_uniform():
    assert entropy(np.ones(4) / 4) == pytest.approx(np.log(4))


def test_info_gain_perfect_split():
    assert info_gain(np.array([.5, .5]), np.array([1., 0.])) == pytest.approx(np.log(2), abs=1e-6)


def test_info_gain_useless():
    assert info_gain(np.array([.5, .5]), np.array([.7, .7])) == pytest.approx(0, abs=1e-9)


def test_update_yes_no_unknown():
    pr, lik = np.array([.5, .5]), np.array([.9, .1])
    assert update(pr, lik, "yes") == pytest.approx([.9, .1])
    assert update(pr, lik, "no") == pytest.approx([.1, .9])
    assert update(pr, lik, "unknown") == pytest.approx(pr)


def test_candidate_prior_drops_small_third():
    p = np.array([.5, .45, .03, .02])
    idx, prior = candidate_prior(p)
    assert list(idx) == [0, 1]
    assert prior == pytest.approx([.5 / .95, .45 / .95])


def test_candidate_prior_keeps_three():
    idx, prior = candidate_prior(np.array([.1, .4, .3, .2]))
    assert list(idx) == [1, 2, 3]
    assert prior.sum() == pytest.approx(1)


# 3 classes; s0 useless, s1 separates class 0 from 1, s2 weak
L = np.array([[.5, .5, .5],
              [.95, .05, .5],
              [.6, .4, .5]])


def test_not_gated_when_low_entropy():
    st = start(np.array([.98, .01, .01]), H=0.1, tau=0.5)
    assert next_question(st, L) is None
    c, p, confirmed = final(st)
    assert (c, confirmed) == (0, True)


def test_ig_picks_most_separating():
    st = start(np.array([.5, .5, 0.]), H=1.0, tau=0.5)
    assert next_question(st, L) == 1


def test_stops_at_085():
    st = start(np.array([.5, .5, 0.]), H=1.0, tau=0.5)
    st = answer(st, 1, "yes", L)
    assert is_done(st)
    assert final(st)[0] == 0 and final(st)[2]


def test_all_unknown_terminates_unconfirmed():
    pred, n = run_dialogue(np.array([.5, .5, 0.]), 1.0, 0.5, L, lambda s: "unknown")
    assert n <= 3
    st = start(np.array([.5, .5, 0.]), H=1.0, tau=0.5)
    for _ in range(3):
        s = next_question(st, L)
        if s is None:
            break
        st = answer(st, s, "unknown", L)
    assert next_question(st, L) is None
    assert final(st)[2] is False


def test_random_strategy_uses_rng_and_never_repeats():
    L5 = np.random.default_rng(1).uniform(.1, .9, size=(6, 3))
    st = start(np.array([.4, .35, .25]), H=1.0, tau=0.5)
    rng = np.random.default_rng(0)
    seen = []
    while (s := next_question(st, L5, strategy="random", rng=rng)) is not None:
        assert s not in seen
        seen.append(s)
        st = answer(st, s, "unknown", L5)
    assert len(seen) == 3
