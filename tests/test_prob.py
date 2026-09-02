import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from claraenc.ProbCoder import ProbModel, parse_pattern, predict_masked


# bits (MSB-first): 110, 111, 000, 001
# col0 == col1 (perfect correlation), col2 independent-ish, mu all 0.5
TINY = np.array([6, 7, 0, 1], dtype=np.uint8)


@pytest.fixture
def tiny():
    return ProbModel(TINY, eps=1e-4)


def test_parse_pattern():
    assert parse_pattern("0101") == ([0, 1, 2, 3], [0, 1, 0, 1])
    assert parse_pattern("__1____") == ([2], [1])
    assert parse_pattern("_. .0") == ([4], [0])
    with pytest.raises(ValueError):
        parse_pattern("01x0")


def test_predict_pattern_too_long(tiny):
    with pytest.raises(ValueError):
        tiny.predict("1111")


def test_fit_w_hand(tiny):
    # target col1 on [c0, c2]: c1 == c0 exactly -> coefs [1, 0]
    assert tiny.W[0, 1] == pytest.approx(1.0)
    assert tiny.W[2, 1] == pytest.approx(0.0)
    # target col0 symmetric
    assert tiny.W[1, 0] == pytest.approx(1.0)
    # target col2 on [c0, c1]: c2 uncorrelated with both -> zero influence
    assert tiny.W[0, 2] == 0
    assert tiny.W[1, 2] == 0
    assert np.all(np.diag(tiny.W) == 0)


def test_predict_defined_exact(tiny):
    p = tiny.predict("1__")
    assert p[0] == 1.0
    assert p[1] == pytest.approx(1.0, abs=1e-3)   # perfectly correlated
    assert p[2] == pytest.approx(0.5)            # col2 uncorrelated with col0


def test_predict_idx_matches_pattern(tiny):
    assert np.allclose(tiny.predict("1__"), tiny.predict_idx([(0, 1)]))


def test_empirical_probs(tiny):
    p = tiny.empirical_probs([0])
    # bit0=1 group (items 6,7): col1 mean 1, col2 mean 0.5
    assert p[0, 0] == 1.0
    assert p[0, 1] == pytest.approx(1.0, abs=1e-3)
    assert p[1, 1] == pytest.approx(1.0, abs=1e-3)
    assert p[0, 2] == 0.5
    assert p[1, 2] == 0.5
    # bit0=0 group (items 0,1): col1 mean 0
    assert p[2, 1] == pytest.approx(0.0, abs=1e-3)
    assert p[3, 1] == pytest.approx(0.0, abs=1e-3)


def test_total_bits_hand(tiny):
    # all means 0.5 -> 4 items * 3 bits * 1 bit
    assert tiny.total_bits([]) == pytest.approx(12.0)
    # define bit 0: 4 raw bits + col1 (exact, ~eps clip) + col2 (p=0.5)
    expected = 4 + 4 * -np.log2(1 - tiny.eps) + 4 * 1
    assert tiny.total_bits([0]) == pytest.approx(expected)


def test_greedy_select(tiny):
    picks, s = tiny.greedy_select()
    # bit 0 pays (correlation with col1), bit 2 is uncorrelated and
    # defining more raw bits only costs -> stops after one pick
    assert len(picks) == 1
    assert picks[0][0] == 0
    assert picks[0][1] == pytest.approx(12.0 - tiny.total_bits([0]))
    assert s == [0]


def test_constant_bit_zero_influence():
    # 14=1110, 15=1111, 8=1000, 9=1001: col0 constant -> W row 0 exact 0
    m = ProbModel(np.array([14, 15, 8, 9], dtype=np.uint8), eps=1e-4)
    assert np.all(m.W[0] == 0)
    p_def = m.predict("1___")
    p_free = m.predict("____")
    assert p_def[0] == 1.0
    assert np.array_equal(p_def[1:], p_free[1:])


def test_loo_runs(tiny):
    total = tiny.loo_bits([0])
    assert np.isfinite(total) and total > 0


def test_print_evaluation_runs(tiny, capsys):
    tiny.print_evaluation([0])
    out = capsys.readouterr().out
    assert "==Evaluate== fit=full S=[0]" in out
    assert "error rate" in out
    assert "x_S=1" in out and "x_S=0" in out


def test_predict_masked():
    x = np.array([1.0, -1.0, 0.5])
    defined = np.array([1, 1, 0])
    w = np.array([[0.0, 1.0],
                  [2.0, 0.0],
                  [9.0, 9.0]])
    # (x * defined) @ w: row 3 of x masked out -> [[-2, 1]]
    assert np.allclose(predict_masked(x, defined, w), [[-2.0, 1.0]])


def test_centered_query():
    # 9 = 1001 (x7), 0 = 0000 (x2): col0 7/9 ones, col3 == col0 except one
    # 8 = 1000 breaks the perfect correlation; col0 mean is far from 0
    data = np.array([9] * 6 + [8] + [0, 0], dtype=np.uint8)
    m = ProbModel(data)
    # full fit on centered columns: prediction must use (x - mu)
    # bit0=1 -> empirical P(col3=1) = 6/7; bit0=0 -> 0/2 = 0
    assert m.predict_idx([(0, 1)])[3] == pytest.approx(6 / 7)
    assert m.predict_idx([(0, 0)])[3] == pytest.approx(0.0, abs=1e-3)
    emp = m.empirical_probs([0])
    assert emp[m.bits[:, 0] == 1, 3].mean() == pytest.approx(6 / 7)


def test_fit_single_exact_single_bit():
    # fit='single' = one-bit-at-a-time: for |S| = 1 the prediction hits
    # the empirical conditional mean EXACTLY (2-point line)
    data = np.array([9] * 6 + [8] + [0, 0], dtype=np.uint8)
    m = ProbModel(data, fit="single")
    assert m.predict_idx([(0, 1)])[3] == pytest.approx(6 / 7)
    assert m.predict_idx([(0, 0)])[3] == pytest.approx(0.0, abs=1e-3)


def test_fit_single_vs_full_identical_predictors():
    # 7 = 111, 0 = 000: all three columns identical
    data = np.array([7, 7, 0, 0], dtype=np.uint8)
    full = ProbModel(data)
    single = ProbModel(data, fit="single")
    # full: target col1 on two identical predictors -> min-norm 0.5/0.5
    assert full.W[0, 1] == pytest.approx(0.5)
    # single: cov/var per row -> 1.0
    assert single.W[0, 1] == pytest.approx(1.0)
    assert full.predict_idx([(0, 1)])[1] == pytest.approx(0.75)
    assert single.predict_idx([(0, 1)])[1] == pytest.approx(1.0, abs=1e-3)


def test_fit_unknown_raises():
    with pytest.raises(ValueError):
        ProbModel(np.array([6, 7, 0, 1], dtype=np.uint8), fit="bogus")


def test_hermes_smoke(hermes_weights):
    m = ProbModel(hermes_weights)
    picks, s = m.greedy_select()
    assert len(picks) >= 1
    assert all(gain > 0 for _, gain in picks)
    cur = m.total_bits([])
    for pos in s:
        cur = m.total_bits(s[: s.index(pos) + 1])
    assert m.total_bits(s) <= m.total_bits([])


def test_fit_raw_equals_full_when_mean_zero(tiny):
    # TINY has mu = 0 everywhere -> centering is a no-op, raw == full
    raw = ProbModel(TINY, fit="raw")
    assert np.allclose(raw.W, tiny.W)
    assert np.allclose(raw.predict_idx([(0, 1)]), tiny.predict_idx([(0, 1)]))


def test_fit_raw_no_mean_at_query():
    # fit='raw' never adds the mean: an empty query has no offset at
    # all, every bit comes out exactly 0.5
    m = ProbModel(np.array([9] * 6 + [8] + [0, 0], dtype=np.uint8), fit="raw")
    p = m.predict("___")
    assert np.allclose(p, 0.5)
    # defining the constant bit 2 at its true value 0 re-enters a
    # column that carries part of the dataset offset -> non-trivial p
    p2 = m.predict("__0_")
    assert not np.allclose(p2, 0.5)
    assert np.all((p2[[0, 3]] >= m.eps) & (p2[[0, 3]] <= 1 - m.eps))


def test_hermes_raw_fit_smoke(hermes_weights):
    m = ProbModel(hermes_weights, fit="raw")
    picks, s = m.greedy_select()
    assert all(gain > 0 for _, gain in picks)
    assert np.isfinite(m.loo_bits(s))


def test_hermes_single_fit_smoke(hermes_weights):
    m = ProbModel(hermes_weights, fit="single")
    picks, s = m.greedy_select()
    assert len(picks) >= 1
    assert all(gain > 0 for _, gain in picks)
    assert np.isfinite(m.loo_bits(s))
