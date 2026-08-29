from collections import Counter
from enum import StrEnum
from typing import NamedTuple, List

import numpy as np
from pathlib import Path

# from BitWriter import BitWriter
from clarastrings import FramePrint
from clarastrings import BeginItemOptions, ParentChildRelation, ItemClosingBeavior
from clarautils import Bitty, NBitArray, SliceView, build_bins_n_print
from clarautils import get_bits, get_number, symbol_to_str, get_bitmask, get_indices
from entropy import get_bitwise_entropy


def get_bit_count(value: int):
    return int(np.ceil(np.log2(value + 1)))


class Char(StrEnum):
    fill = "."
    down = "↧"
    branch = "X"
    space = " "


def safe_iter(iter, default):
    try:
        return next(iter)
    except StopIteration:
        return default


class DefineBitOp(NamedTuple):
    idx: int
    bit: 1 | 0

    def __repr__(s):
        return f"op:{s.idx}={s.bit}"


class RunOp(NamedTuple):
    abs_pos: int
    bit: 1 | 0
    kind: str
    level: int


class StraitDef(NamedTuple):
    abs_pos: int
    bit: 1 | 0
    determined: str


class EntropyDiff(NamedTuple):
    entropy_before: np.array
    entropy_after: np.array
    idx: int

    @staticmethod
    def _entropy_base_10_str(val):
        ceiling = np.array(np.ceil(val * 9), dtype=np.int32)
        min = np.min((ceiling, np.full_like(ceiling, fill_value=9)), axis=0)
        e_str = symbol_to_str(min)
        return e_str

    def adjust_entropy_after(s, entr_str):
        return "".join([(Char.space if i == s.idx else '') + c for i, c in enumerate(entr_str)])

    def get_before_str(s):
        return s._entropy_base_10_str(s.entropy_before)

    def get_after_str(s):
        result = s._entropy_base_10_str(s.entropy_after)
        return s.adjust_entropy_after(result)

    def get_gain_sum(s):
        return sum(s.entropy_before) - sum(s.entropy_after)


class BuildParams(NamedTuple):
    data: NBitArray
    level: int
    code: str
    value: str
    operation: DefineBitOp | None
    entropy: EntropyDiff | None
    total_bit: int
    remaining_bits: int
    run: tuple

    def get_commons(s):
        level = s.level + 1
        node_name = f"lvl:{level},{s.operation}"
        code = s.code + str(s.operation.bit) if s.level > 0 else ''
        value = "".join([str(s.operation.bit) if c == Char.branch
                         else c
                         for c in s.value])
        return BuildParamsCommons(level, node_name, code, value)

    def create_child(self, data, value, operation, entropy, keep_code=False, abs_op=None):
        bit = operation.idx
        return BuildParams(data=data,
                           level=self.level + 1,
                           code=self.code if keep_code else self.code + str(bit),
                           value=value,
                           operation=operation,
                           entropy=entropy,
                           total_bit=self.total_bit,
                           remaining_bits=self.remaining_bits - 1,
                           run=self.run if abs_op is None else self.run + (abs_op,))
        pass

    @classmethod
    def make_root(cls, data, bit_count):
        return BuildParams(data=data,
                           level=0,
                           code='',
                           value=Char.fill * bit_count,
                           operation=None,
                           entropy=None,
                           total_bit=bit_count,
                           remaining_bits=bit_count,
                           run=())
        pass


class BuildParamsCommons(NamedTuple):
    level: int
    node_name: str
    code: str
    value: np.array


class Split(NamedTuple):
    bit_idx: int
    gains: np.ndarray
    entropy_begin: np.ndarray
    entropy_after_with: np.ndarray
    entropy_after_wout: np.ndarray


class DataStrait(NamedTuple):
    operations: List[DefineBitOp]
    next_node: NamedTuple | None | np.generic


class Node(NamedTuple):
    bit_idx: int | None
    true_node: NamedTuple | None | np.generic
    false_node: NamedTuple | None | np.generic


class GainCoder:
    def merge_str(s, a, b):
        iter_b = iter(b)
        chars = [safe_iter(iter_b, 'E') if a_i == Char.fill else a_i for a_i in a]
        return "".join(chars)

    def __init__(self, values, counts, bit_count):
        self.bit_count = np.uint32(bit_count)
        self.values = np.array(values)
        self.counts = np.array(counts)
        self.mgr = FramePrint().get_mgr()
        self.mgr.print_realtime = False
        self.abs_straits = []
        self.runs = []
        self.node, self.avg_bits, self.leaf_ext, self.bins_ext, self.codes = self._build()

    def get_avg_bits(self):
        return self.avg_bits

    def get_next_split(s, view: NBitArray, bit_count):
        data = view.get_array()
        begin_entropy_list = get_bitwise_entropy(data, bit_count)
        begin_entropy = sum(begin_entropy_list)
        max_g = 0
        max_with_entro = []
        max_wout_entro = []
        max_bit_idx = 0
        increases = []
        for i in range(bit_count):
            groups = view.group_by_bit(bit_count - 1 - i)
            with_parts = groups[1].get_array()
            wout_parts = groups[0].get_array()
            with_e_list = get_bitwise_entropy(with_parts, bit_count - 1)
            wout_e_list = get_bitwise_entropy(wout_parts, bit_count - 1)
            with_g = begin_entropy - np.sum(with_e_list)
            wout_g = begin_entropy - np.sum(wout_e_list)
            gain = (with_g * len(with_parts) + wout_g * len(wout_parts)) / (len(with_parts) + len(wout_parts))
            if gain > max_g:
                max_with_entro = with_e_list
                max_wout_entro = wout_e_list
                max_bit_idx = i
                max_g = gain
            increases.append(gain)
        return Split(bit_idx=max_bit_idx,
                     gains=np.array(increases),
                     entropy_begin=begin_entropy_list,
                     entropy_after_with=max_with_entro,
                     entropy_after_wout=max_wout_entro)

    def _build(self):
        depths = []
        leaf_len = []
        flag_len = []
        codes = {}
        node_count = 0
        strait_count = 0
        leaf_count = 0

        def bit_str(idx, bit: int | str, l: int):
            chars = [Char.fill if i != idx
                     else str(bit) if isinstance(bit, int)
            else bit
                     for i in reversed(range(l))]
            return "".join(chars)

        def get_diff(string_a, sting_b):
            return "".join([Char.down if a != b
                            else Char.space
                            for a, b in zip(string_a, sting_b)])

        def get_abs_positions(view: SliceView) -> List[int]:
            return get_indices(view.bit_slice, int(self.bit_count))

        def get_defined_ops(view: NBitArray) -> List[DefineBitOp]:
            bit_count = view.get_bit_count()
            bits = view.get_bitwise()
            mins = np.min(bits, axis=0)
            maxs = np.max(bits, axis=0)
            return [DefineBitOp(bit_count - 1 - col, 1 if mins[col] > 0 else 0)
                    for col in range(bit_count) if mins[col] == maxs[col]]

        def check_defined(params: BuildParams, value: str | None = None):
            nonlocal leaf_len, strait_count
            value = params.value if value is None else value

            defined = get_defined_ops(params.data)
            if len(defined) == 0:
                return []

            result = []
            entry_bits = params.remaining_bits
            abs_positions = get_abs_positions(params.data)

            for op in defined:
                abs_pos = int(abs_positions[entry_bits - 1 - op.idx])
                self.abs_straits.append(StraitDef(abs_pos, op.bit, value))
                value = self.merge_str(value, bit_str(op.idx, bit=op.bit, l=params.remaining_bits))
                data = params.data.rm_b(params.remaining_bits - 1 - op.idx)
                params = params.create_child(data=data,
                                             operation=op,
                                             value=value,
                                             entropy=None,
                                             keep_code=True,
                                             abs_op=RunOp(abs_pos, op.bit, 'strait', params.level + 1))

                flag_len.append(len(get_bits(op.idx)))
                strait_count += 1
                result.append(params)

            if len(get_defined_ops(params.data)) > 0:
                raise Exception("Not all bits are defined")

            return result

        def build_recursive(params: BuildParams):
            if len(params.data) == 1:
                leaf_bits = create_leaf(params)
                return get_number(leaf_bits)

            nonlocal depths, leaf_len, flag_len
            level, node_name, code, value = params.get_commons()

            sb = self.mgr.begin_item(node_name,
                                     options=BeginItemOptions(parent=ParentChildRelation.DirectParentIsParent))
            sb = sb.append(f"({node_name})[in: ")
            offset = sb.get_cursor()

            sb.a(f"{params.value},→ entropy [in: ")
            offset2 = sb.get_cursor()

            entropy = params.entropy

            sb = sb.a(f"{entropy.get_before_str()},").make_next_line() \
                .fill_to(end="changes: ", to=offset).a(f"{get_diff(params.value, value)},") \
                .fill_to(to=offset2).a(
                f"{get_diff(params.entropy.get_before_str(), entropy.get_after_str())},").make_next_line() \
                .fill_to(end="node: ", to=offset).a(f"{value},→") \
                .fill_to(end="etp node: ", to=offset2).a(
                f"{entropy.get_after_str()}] gain:{entropy.get_gain_sum():.3f}").make_next_line()

            for p in check_defined(params, value):
                sb = sb.fill_to(end=f"{p.operation}: ", to=offset).a(f"{get_diff(value, p.value)},").make_next_line() \
                    .fill_to(end="now: ", to=offset).a(f"{p.value},").make_next_line()
                params = p
                value = p.value

            split = self.get_next_split(params.data, params.remaining_bits)
            out_val = self.merge_str(value, bit_str(split.bit_idx, bit=Char.branch, l=params.remaining_bits))

            sb.fill_to(end=f"op:{split.bit_idx}={Char.branch}: ", to=offset).a(
                f"{get_diff(value, out_val)},").make_next_line() \
                .fill_to(end="out: ", to=offset).a(f"{out_val}]").make_next_line()

            node = create_node(out_val, params, split, split.bit_idx)

            sb = self.mgr.close_item(node_name)
            sb.append(f"({node_name}) End")
            return node

        def create_node(out_val, params, split, value: int | None = None):
            nonlocal leaf_len, node_count

            if value is not None:
                flag_len.append(len(get_bits(value)))

            abs_positions = get_abs_positions(params.data)
            split_msb = params.remaining_bits - 1 - split.bit_idx
            abs_pos = int(abs_positions[split_msb])
            split_level = params.level + 1
            groups = params.data.group_by_bit(split_msb)

            true_node = build_recursive(params.create_child(groups[1], out_val,
                                                             operation=DefineBitOp(split.bit_idx, bit=1),
                                                             entropy=EntropyDiff(split.entropy_begin,
                                                                                 split.entropy_after_with,
                                                                                 split.bit_idx),
                                                             abs_op=RunOp(abs_pos, 1, 'split', split_level)))
            false_node = build_recursive(params.create_child(groups[0], out_val,
                                                              operation=DefineBitOp(split.bit_idx, bit=0),
                                                              entropy=EntropyDiff(split.entropy_begin,
                                                                                  split.entropy_after_wout,
                                                                                  split.bit_idx),
                                                              abs_op=RunOp(abs_pos, 0, 'split', split_level)))
            node_count += 1
            return Node(value, true_node, false_node)

        def create_leaf(params):
            nonlocal depths, leaf_len, codes, leaf_count
            data, parent_level, parent_code, parent_value, op, entropy, bit_count, remaining_bits, run = params
            level, node_name, code, value = params.get_commons()
            depths.append(level)
            self.runs.append(run)

            leaf_value = data.get_array()[0]
            leaf_bits = get_bits(leaf_value, remaining_bits)
            leaf_str = symbol_to_str(leaf_bits)
            leaf_len.append(len(get_bits(leaf_value)))
            full = self.merge_str(value, leaf_str)
            sb = self.mgr.begin_item(leaf_value,
                                     options=BeginItemOptions(parent=ParentChildRelation.DirectParentIsParent,
                                                              closing_beavior=ItemClosingBeavior.NoClosingTagFitChildren))
            sb = sb.append(f"({node_name})[in: ")
            offset = sb.get_cursor()
            sb.a(f"{parent_value}, leaf: ")
            offset2 = sb.get_cursor() + 8
            sb.a(f"({leaf_value})-[{leaf_str}]").make_next_line() \
                .fill_to(end="changes: ", to=offset).a(f"{get_diff(parent_value, value)},").fill_to(end="╰→[node: ",
                                                                                                    to=offset2).a(
                f"{value},").make_next_line() \
                .fill_to(end="node: ", to=offset).a(f"{value}]").fill_to(end="changes: ", to=offset2).a(
                f"{get_diff(value, full)},").make_next_line() \
                .fill_to(end="out: ", to=offset2).a(f"{full}]").make_next_line()

            bits = get_bits(full)
            int = get_number(bits)
            codes[int.value] = params.code

            leaf_count += 1
            return leaf_bits

        def make_root(data):
            root = "root"
            sb = self.mgr.begin_item(root, options=BeginItemOptions(parent=ParentChildRelation.DirectParentIsParent))
            sb.append(f"({root})     [")
            offset = sb.get_cursor()

            bit_count = int(self.bit_count)
            params = BuildParams.make_root(data=SliceView(Bitty(data, max_bit=bit_count)),
                                           bit_count=bit_count)

            sb = sb.append(f"{params.value}").make_next_line()

            for p in check_defined(params):
                sb = sb.fill_to(end=f"{p.operation}: ", to=offset).a(
                    f"{get_diff(params.value, p.value)},").make_next_line() \
                    .fill_to(end="now: ", to=offset).a(f"{p.value},").make_next_line()
                params = p

            split = self.get_next_split(params.data, params.remaining_bits)
            out_val = self.merge_str(params.value, bit_str(split.bit_idx, bit=Char.branch, l=params.remaining_bits))

            sb.fill_to(end="changes: ", to=offset).a(
                f"{get_diff(params.value, out_val)}, gain:{np.max(split.gains)}").make_next_line() \
                .fill_to(end="out: ", to=offset).a(f"{out_val}]")

            node = create_node(out_val, params, split)

            sb = self.mgr.close_item(root)
            sb.append(f"({root}) End")

            return node

        tree = make_root(self.values)

        bit_count = int(self.bit_count)
        self.strait_levels = [[] for _ in range(bit_count)]
        self.split_levels = [[] for _ in range(bit_count)]
        for run in self.runs:
            for op in run:
                if op.kind == 'strait':
                    self.strait_levels[op.abs_pos].append(op.level)
                else:
                    self.split_levels[op.abs_pos].append(op.level)

        self.abs_strait_pos = np.array([s.abs_pos for s in self.abs_straits], dtype=np.uint32)
        self.abs_split_pos = np.array([op.abs_pos for r in self.runs for op in r if op.kind == 'split'],
                                      dtype=np.uint32)

        print("==Data==")
        print("Codes:", len(codes), "Nodes:", node_count, "Straits:", strait_count, "Leafs:", leaf_count)
        print("==Leafs==")
        leaf_ext_delta = build_bins_n_print(leaf_len, [45, 90, 100])
        print("==Flags==")
        bins_ext_delta = build_bins_n_print(flag_len, [45, 90, 100])
        print("==AbsStraits==")
        s_counts = np.zeros((2, bit_count), dtype=np.uint32)
        for s in self.abs_straits:
            s_counts[s.bit, s.abs_pos] += 1
        build_bins_n_print(self.abs_strait_pos, [45, 90, 100])
        strait_rules = Counter(((int(s.abs_pos), int(s.bit)) for s in self.abs_straits))
        print(f"Rules: {len(strait_rules)} unique of {len(self.abs_straits)}")
        print("Top:", ", ".join(f"{p}={b}×{c}" for (p, b), c in strait_rules.most_common(5)))
        print("==AbsSplits==")
        build_bins_n_print(self.abs_split_pos, [45, 90, 100])

        return tree, np.average(depths), leaf_ext_delta, bins_ext_delta, codes

    def average_bits(self):
        total = sum(self.counts)
        return sum(len(self.codes[v]) * c for v, c in zip(np.array(self.values), self.counts)) / total

    def compress_tree(self):
        # bw = BitWriter()
        # layer = [self.node.true_node, self.node.false_node]
        # while len(layer) > 0:
        #     for n in layer:
        #         # if isinstance(n, np.generic):
        #         #     bw.put(False)
        #         #     bw.put(n, length=self.leaf_ext)
        #         # else:
        #         #     bw.put(True)
        #         n.bit
                pass

    def compression_ratio(self, original_bits=16):
        return self.average_bits() / original_bits

    def print(self):
        print(self.mgr)

def plot_bit_definition_order(coder):
    strait_qs = np.array([np.percentile(p, [25, 50, 75]) if len(p) > 0 else [0, 0, 0]
                          for p in coder.strait_levels])
    split_qs = np.array([np.percentile(p, [25, 50, 75]) if len(p) > 0 else [0, 0, 0]
                         for p in coder.split_levels])
    strait_order = np.argsort(strait_qs.mean(axis=1))
    split_order = np.argsort(split_qs.mean(axis=1))
    print(f"AbsStraits sorted (pos, [q25, q50, q75]): "
          f"{[(int(p), q.tolist()) for p, q in zip(strait_order, strait_qs[strait_order])]}")
    print(f"AbsSplits sorted (pos, [q25, q50, q75]): "
          f"{[(int(p), q.tolist()) for p, q in zip(split_order, split_qs[split_order])]}")

    import matplotlib.pyplot as plt
    fig, (ax_straits, ax_splits) = plt.subplots(2, 1, figsize=(12, 8))
    for ax, levels, order, title in ((ax_straits, coder.strait_levels, strait_order, "Strait levels per abs bit pos"),
                                     (ax_splits, coder.split_levels, split_order, "Split levels per abs bit pos")):
        ax.boxplot([levels[p] for p in order])
        ax.set_xticks(np.arange(1, len(order) + 1), [int(p) for p in order])
        ax.set_title(title)
        ax.set_xlabel("abs bit pos")
        ax.set_ylabel("level")
    fig.tight_layout()
    fig.savefig("levels_boxplot.png")
    plt.show()

def plot_strait_counts(coder):
    bit_count = int(coder.bit_count)
    s_counts = np.zeros((2, bit_count), dtype=np.uint32)
    for s in coder.abs_straits:
        s_counts[s.bit, s.abs_pos] += 1
    print(f"Strait count bins (pos: bit0/bit1): "
          f"{[(p, int(s_counts[0, p]), int(s_counts[1, p])) for p in range(bit_count) if s_counts[:, p].any()]}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(bit_count)
    width = 0.4
    ax.bar(positions - width / 2, s_counts[0], width, label="bit=0")
    ax.bar(positions + width / 2, s_counts[1], width, label="bit=1")
    ax.set_title("Strait counts per abs bit pos and bit value")
    ax.set_xlabel("abs bit pos")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig("strait_counts.png")
    plt.show()

base = Path("F:\\source\\sandbox314\\modelCompression\\bins")

bits_to_shift = 0
bits_to_take = 32
mask = get_bitmask(bits_to_take)
num_possible = np.pow(2, bits_to_take)
# for i in range(1):
for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
    with open(path, "rb") as f:
        buffer = f.read()
    name = path.name

    x = np.frombuffer(buffer, dtype=np.uint32)

    values, counts = np.unique(x, return_counts=True)

    values = values.view()

    num_possible = np.iinfo(np.uint32).max + 1
    num_unique = len(values)
    ratio = num_unique / num_possible

    bit_req = get_bit_count(num_unique)
    print(f"{name}: {num_unique}({bit_req:.3f} bits) unique, ratio={ratio:.6f}")

    coder = GainCoder(values, counts, bits_to_take)

    avg_bits = coder.average_bits()
    ratio_bits = coder.compression_ratio(bits_to_take)

    plot_bit_definition_order(coder)
    plot_strait_counts(coder)


    # total = np.sum(counts)
    # probabilities = counts / total
    #
    # # # Ignore zero probabilities to avoid log2(0) warning
    # # valid_probs = probabilities[probabilities > 0]
    #
    # # Shannon Entropy = expected average bits per symbol for AC
    # entropy_avg_bits = -np.sum(probabilities * np.log2(probabilities))
    #
    # print(f"AC theoretical avg bits: {entropy_avg_bits:.3f}")
    # print(f"Huffman actual avg bits: {coder.average_bits():.3f}")
    #
    # # plot_code_length_histogram(coder, path.name)
