import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from claraenc.ProbCoder import ProbModel, parse_pattern


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


def test_hermes_smoke(hermes_weights):
    m = ProbModel(hermes_weights)
    picks, s = m.greedy_select()
    assert len(picks) >= 1
    assert all(gain > 0 for _, gain in picks)
    cur = m.total_bits([])
    for pos in s:
        cur = m.total_bits(s[: s.index(pos) + 1])
    assert m.total_bits(s) <= m.total_bits([])
