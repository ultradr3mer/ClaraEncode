from itertools import permutations, islice, combinations
from pathlib import Path

import numpy as np

from claraenc.sandbox_paths import sandbox_path
from clarautils import get_bits


def transformationsmatrix(x, y):
    """
    Erzeugt eine Matrix M, sodass gilt:
        x @ M = y

    x und y sind Zeilenvektoren gleicher Länge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x und y müssen eindimensionale Vektoren sein.")

    if np.allclose(x, 0):
        raise ValueError("Der Vektor x darf nicht der Nullvektor sein.")

    # M erfüllt x @ M = y
    M = np.outer(x, y) / np.dot(x, x)
    return M

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def generate_spread_sets(candidates, count=None, return_distances=False):
    """
    Spread-Set-Generator: wählt aus `candidates` (2-D, Zeilen = Vektoren)
    gierig Zeilen mit maximaler Hamming-Distanz zur bereits gewählten
    Menge und yielded sie einzeln — lazy, Ebene für Ebene.

    Zeile 0 ist der Startpunkt; danach bilden pro Runde alle Kandidaten
    mit maximaler Mindestdistanz die aktuelle Ebene, und daraus werden
    gierig die behalten, die untereinander mindestens diesen Abstand
    einhalten.

    count=None → alle; sonst Stop nach `count` Zeilen
    (IndexError falls len(candidates) < count).
    return_distances=True → yielded (Zeile, Distanz)-Tupel; Distanz =
    Ebene, in der die Zeile gewählt wurde (Startzeile: None).

    Beispiel — die ersten 10 Zeilen als Array:
        np.array(islice(generate_spread_sets(cands), 10))

    Performance: 0/1-Daten → Hamming über Matrizenmultiplikation
    (hamming(a,b) = |a| + |b| - 2·a·b) statt 3D-Broadcast, und
    Mindestdistanzen werden inkrementell gegen nur die NEUE Auswahl
    aktualisiert statt gegen die ganze gewählte Menge neu gerechnet.
    """
    candidates = np.asarray(candidates)
    if candidates.ndim != 2:
        raise ValueError("candidates muss 2-D sein (items x bits).")
    n = candidates.shape[0]
    if count is not None:
        if n < count:
            raise IndexError(f"Too few candidates: required={count}, candidates={n}")
        if count <= 0 or n == 0:
            return
    elif n == 0:
        return

    binary = (np.issubdtype(candidates.dtype, np.bool_)
              or (np.issubdtype(candidates.dtype, np.integer)
                  and candidates.min() >= 0 and candidates.max() <= 1))
    work = candidates.astype(np.float32) if binary else candidates

    if binary:
        def pair_dist(a, b):
            # Hamming-Distanzen Zeilen a x Zeilen b (0/1-Daten)
            return (a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :]
                    - 2.0 * (a @ b.T))
    else:
        def pair_dist(a, b):
            # korrekt für beliebige Werte (wie die alte Broadcast-Version)
            return np.sum(np.abs(a[:, None, :] - b[None, :, :]), axis=2)

    if return_distances:
        yield candidates[0], None
    else:
        yield candidates[0]
    written = 1

    rem_idx = np.arange(1, n)
    rem = work[1:]
    min_d = pair_dist(rem, work[0:1])[:, 0]

    while rem_idx.size:
        if count is not None and written >= count:
            return
        max_d = min_d.max()
        sub_pos = np.flatnonzero(min_d == max_d)
        if max_d == 0:
            # jeder Abstand ist >= 0 → die ganze Ebene passt
            batch_pos = sub_pos
        else:
            # gierige Auswahl in der Ebene: untereinander >= max_d
            sub = rem[sub_pos]
            alive = np.arange(len(sub_pos))
            alive_min = np.full(len(sub_pos), np.inf)
            picked_local = []
            while alive.size:
                p = alive[0]
                picked_local.append(p)
                rest = alive[1:]
                if rest.size == 0:
                    break
                d = pair_dist(sub[rest], sub[p:p+1])[:, 0]
                rest_min = np.minimum(alive_min[1:], d)
                keep = rest_min >= max_d
                alive = rest[keep]
                alive_min = rest_min[keep]
            batch_pos = sub_pos[np.array(picked_local)]

        batch_rows = candidates[rem_idx[batch_pos]]
        keep_mask = np.ones(rem_idx.size, dtype=bool)
        keep_mask[batch_pos] = False
        if keep_mask.any():
            d = pair_dist(rem[keep_mask], rem[batch_pos]).min(axis=1)
            min_d[keep_mask] = np.minimum(min_d[keep_mask], d)
        rem_idx = rem_idx[keep_mask]
        rem = rem[keep_mask]
        min_d = min_d[keep_mask]

        for row in batch_rows:
            if count is not None and written >= count:
                return
            if return_distances:
                yield row, max_d
            else:
                yield row
            written += 1

def calc_req_counter_bits(item_count: int, defined: int, max=128):
    for bit_count in range(defined+1,max):
        possibilities = np.prod([bit_count - i
                                 for i in range(defined)])
        if possibilities >= item_count:
            return bit_count, defined
    raise ValueError("max bits exceeded")

def random_model(x):
    # x = x[:128]
    item_count = len(x)
    bits = get_bits(x)
    bit_count = bits.shape[1]
    # v_shape = (bits.shape[0], bits.shape[1] * 2)
    # vector = np.zeros(shape=v_shape, dtype=float)
    # vector[:, 0::2] = np.array(bits, dtype=float)
    # vector[:, 1::2] = np.array(bits, dtype=float) - 1
    bits_vec = np.array(bits, dtype=float) * 2 - 1
    # bits_vec = bits_vec - np.mean(bits_vec, axis=0, keepdims=True)

    counter_bit_count, counter_n_defined = calc_req_counter_bits(item_count, defined=4)
    bit_list = list(range(counter_bit_count))
    counter_per = np.array(list(islice(permutations(bit_list,counter_n_defined),item_count)))
    counter_bits = get_bits(np.sum(1 << counter_per, axis=1))
    # spread: counter_bits = np.array(islice(generate_spread_sets(counter_bits), item_count))


    # spreadset = generate_spread_sets(10,4)

    rng = np.random.default_rng(seed=42)
    r_shape = (bits.shape[0], 32)
    rand_b = rng.integers(0, 2, size=r_shape, dtype=np.uint8)
    rand_vec = np.array(rand_b, dtype=float)

    # print(counter_vec)
    # [[-1. -1.  1. ... -1.  1. -1.]
    #  [-1. -1.  1. ...  1.  1.  1.]
    #  [-1. -1.  1. ... -1. -1. -1.]
    #  ...
    #  [-1. -1.  1. ... -1.  1. -1.]
    #  [ 1. -1.  1. ... -1. -1. -1.]
    #  [ 1. -1.  1. ... -1. -1.  1.]]

    # initial_p = np.mean(bits, axis=0)
    # print(initial_p)

    bit_count = bits.shape[1]
    bit_count_expanded = bit_count * 2
    # w = np.zeros(shape=(bit_count_expanded,bit_count_expanded), dtype=float)
    # for j in range(bit_count):
    #     j=j*2
    #     all_j = [j, j+1]
    #     others = list(i for i in range(bit_count_expanded) if i != j and i != j+1)
    #     x = vector[:, others]
    #     y = vector[:, all_j]
    #     w_sub = np.linalg.lstsq(x, y, rcond=None)[0]
    #     print("w_sub:")
    #     print(w_sub)
    #     w[others, j:j+2] = w_sub
    w = np.linalg.lstsq(counter_bits, bits_vec, rcond=None)[0]
    print("w:")
    print(w)

    w_power = np.mean(np.abs(w), axis=0)
    w_max = np.argsort(-w_power)
    w_max_n = w_max[:24]

    y = bits
    approx = counter_bits @ w
    print("y:")
    print(y)
    print("approx:")
    print(approx)
    print("err:")
    approx_bits = np.where(approx>0,1,0)
    err = np.abs(approx_bits - bits)
    print(err)
    print(f"total: {np.sum(err)}/{err.size}")


if __name__ == "__main__":
    base = sandbox_path("bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        x = np.frombuffer(buffer, dtype=np.uint16)

        random_model(x)
