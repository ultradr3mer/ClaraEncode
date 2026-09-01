"""
ProbCoder — conditional bit probabilities from a global influence matrix.

Idea (Clara): start from the general per-bit probability (initial mean),
then define as few bits as possible; the defined bits adjust the
probabilities of the remaining (undefined) bits. Linear model in +/-1
space => regression on centered data:

    p_j(x_S) = clip((1 + mu_j + sum_{i in S} W[i, j] * x_i) / 2, eps, 1 - eps)

with X = 2*bits - 1, mu = X.mean(axis=0) (mu + 1)/2 = initial_p,
W (bit_count x bit_count, diagonal 0) fitted per target column j via
least squares of Xc[:, others] -> Xc[:, j].

Tradeoff of the global matrix (chosen over refit-per-query): W is fitted
with ALL other bits present, so a subset query sums marginal influences
(a slight bias). `empirical_probs` gives the exact conditional reference,
`loo_bits` the honest out-of-sample cost.

Bit convention: MSB-first, pattern index i = bit position i (0 = MSB).
Defined bits cost 1 raw bit per item; undefined bits are coded with p.
"""

from pathlib import Path
import sys

import numpy as np
from clarautils import get_bits

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.exampe_data import hermes_weights_data

UNDEFINED_CHARS = "_. "


def parse_pattern(pattern):
    """
    '0'/'1' = defined bit at position i, '_'/'.'/' ' = undefined.
    Short patterns are left-aligned, the rest counts as undefined.

    Returns (positions, values).
    """
    positions = []
    values = []
    for i, c in enumerate(pattern):
        if c in "01":
            positions.append(i)
            values.append(int(c))
        elif c in UNDEFINED_CHARS:
            continue
        else:
            raise ValueError(f"invalid pattern char {c!r} at position {i}")
    return positions, values


def fit_w(xc):
    """
    Influence matrix W from centered +/-1 data (n, bit_count):
    per target column j least squares over all OTHER columns.
    Constant columns center to zeros -> min-norm solution 0; sub-1e-12
    lstsq noise is cleaned to exact 0 (a constant bit has no influence).
    """
    bit_count = xc.shape[1]
    w = np.zeros((bit_count, bit_count))
    idx = np.arange(bit_count)
    for j in range(bit_count):
        others = idx[idx != j]
        coef, *_ = np.linalg.lstsq(xc[:, others], xc[:, j], rcond=None)
        w[others, j] = coef
    w[np.abs(w) < 1e-12] = 0
    return w


class ProbModel:
    def __init__(self, data, eps=1e-4):
        self.data = np.asarray(data)
        self.eps = float(eps)
        self.bits = get_bits(self.data).astype(np.int8)      # (n, bit_count) 0/1
        self.n, self.bit_count = self.bits.shape
        self.x = self.bits.astype(float) * 2 - 1             # +/-1
        self.mu = self.x.mean(axis=0)                        # +/-1 mean
        self.initial_p = (self.mu + 1) / 2                   # = bits.mean(axis=0)
        self.xc = self.x - self.mu
        self.W = fit_w(self.xc)

    def predict(self, pattern):
        """
        Probability vector (bit_count,) for a pattern like '10110_0100'
        or '__1____'. Defined positions return exact 0/1.
        """
        positions, values = parse_pattern(pattern)
        if len(pattern) > self.bit_count:
            raise ValueError(
                f"pattern has {len(pattern)} chars, data only {self.bit_count} bits")
        p = self.predict_idx(list(zip(positions, values)))
        return p

    def predict_idx(self, defined):
        """
        defined: list of (position, value) pairs -> probability vector.
        """
        defined = list(defined)
        adj = sum(self.W[pos] * (val * 2 - 1) for pos, val in defined)
        yhat = self.mu + adj
        p = np.clip((1 + yhat) / 2, self.eps, 1 - self.eps)
        for pos, val in defined:
            p[pos] = val
        return p

    def prob_matrix(self, positions):
        """
        (n, bit_count) probabilities when the given positions are
        defined per item (each item transmits its own values there).
        Undefined columns use the linear model with the item's values.
        """
        positions = list(positions)
        adj = self.x[:, positions] @ self.W[positions, :] if positions \
            else np.zeros((self.n, self.bit_count))
        yhat = self.mu + adj
        p = np.clip((1 + yhat) / 2, self.eps, 1 - self.eps)
        p[:, positions] = self.bits[:, positions]
        return p

    def _code_bits(self, p, bits):
        """
        Cross-entropy in bits: -log2(p) where the bit is 1, -log2(1-p)
        where it is 0. Clip guards against exactly-wrong 0/1 predictions.
        """
        p = np.clip(p, self.eps, 1 - self.eps)
        return -(bits * np.log2(p) + (1 - bits) * np.log2(1 - p))

    def total_bits(self, positions, probs=None):
        """
        Cost of coding the whole dataset with the defined positions:
        |S| * n raw bits + cross-entropy of the undefined bits.
        """
        positions = list(positions)
        if probs is None:
            probs = self.prob_matrix(positions)
        mask = np.ones(self.bit_count, bool)
        mask[positions] = False
        return (len(positions) * self.n
                + self._code_bits(probs[:, mask], self.bits[:, mask]).sum())

    def empirical_probs(self, positions):
        """
        Exact conditional reference: group items by their values on the
        defined positions, per-group per-bit mean. Feasible while
        2**|S| << n.
        """
        positions = list(positions)
        keys = self.bits[:, positions]
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        means = np.array([self.bits[inv == g].mean(axis=0)
                          for g in range(len(uniq))])
        p = np.clip(means[inv], self.eps, 1 - self.eps)
        p[:, positions] = self.bits[:, positions]
        return p

    def greedy_select(self, max_bits=None):
        """
        Forward selection: repeatedly add the position with the biggest
        total_bits reduction (the raw-bit cost of defining is included).
        Stops when no candidate saves anything. Returns (picks, S) with
        picks = [(position, gain), ...].
        """
        picks = []
        s = []
        cur = self.total_bits(s)
        while max_bits is None or len(s) < max_bits:
            best_pos = None
            best_new = None
            best_gain = 0.0
            for pos in range(self.bit_count):
                if pos in s:
                    continue
                new = self.total_bits(s + [pos])
                gain = cur - new
                if gain > best_gain:
                    best_pos, best_new, best_gain = pos, new, gain
            if best_pos is None:
                break
            s.append(best_pos)
            picks.append((best_pos, best_gain))
            cur = best_new
        return picks, s

    def loo_bits(self, positions):
        """
        Leave-one-out total bits for the defined positions: per fold the
        matrix W is refitted on n-1 items, the held-out item is predicted
        with its own defined values. Honest companion to total_bits.
        """
        positions = list(positions)
        mask = np.ones(self.bit_count, bool)
        mask[positions] = False
        total = len(positions) * self.n
        for m in range(self.n):
            train = np.ones(self.n, bool)
            train[m] = False
            mu = self.x[train].mean(axis=0)
            xc = self.x[train] - mu
            w = fit_w(xc)
            adj = sum(w[pos] * self.x[m, pos] for pos in positions)
            yhat = mu + adj
            p = (1 + yhat) / 2
            total += self._code_bits(p[mask], self.bits[m, mask]).sum()
        return total


def fmt_probs(p):
    return " ".join(f"{v:.2f}" for v in p)


if __name__ == "__main__":
    data = hermes_weights_data
    model = ProbModel(data)

    print("initial_p:")
    print(fmt_probs(model.initial_p))

    first8 = "".join(str(b) for b in model.bits[0][:8])
    demo_patterns = ["__1____", first8]
    for pattern in demo_patterns:
        p = model.predict(pattern)
        print(f"predict({pattern!r}):")
        print(fmt_probs(p))

    picks, s = model.greedy_select()
    print("==Selection==")
    for pos, gain in picks:
        print(f"bit {pos:2d}  gain {gain:8.2f} bits")
    print(f"S = {s}")

    print("==Cost== (total bits / per item)")
    independent = model.total_bits([])
    linear = model.total_bits(s)
    empirical = model.total_bits(s, model.empirical_probs(s))
    loo = model.loo_bits(s)
    n = model.n
    print(f"independent : {independent:10.2f} / {independent / n:6.3f}")
    print(f"linear W    : {linear:10.2f} / {linear / n:6.3f}")
    print(f"empirical   : {empirical:10.2f} / {empirical / n:6.3f}")
    print(f"LOO linear  : {loo:10.2f} / {loo / n:6.3f}")
