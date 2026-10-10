from enum import StrEnum
from typing import NamedTuple, List

import numpy as np
from pathlib import Path

# from BitWriter import BitWriter
from clarastrings import FramePrint
from clarastrings import BeginItemOptions, ParentChildRelation, ItemClosingBeavior
from clarautils.commonEncoding import get_bits, get_number, symbol_to_str, get_bitmask
from claraenc.entropy import get_bitwise_entropy


def get_bit_count(value: int):
    return int(np.ceil(np.log2(value + 1)))


def divide(ary, b):
    # if not isinstanceof(ary, np.ndarray):
    #     ary = np.array(ary)
    w = np.where(ary & b > 0, ary, 0)
    with_i = np.nonzero(w)
    wout_i = np.nonzero(w == 0)
    return ary[with_i], ary[wout_i]


def get_without_bit(ary, bit_index):
    before = ary & get_bitmask(length=bit_index)
    after = ary - (ary & get_bitmask(length=bit_index + 1))
    return before + (after >> 1)


def divide_without(ary, bit_index):
    with_parts, wout_parts = divide(ary, np.pow(2, bit_index))

    return get_without_bit(with_parts, bit_index), get_without_bit(wout_parts, bit_index)


# # test = np.array((get_number([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]),
# #                  get_number([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]),
# #                  get_number([1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1]),
# #                  get_number([1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1]),
# #                  get_number([0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1]),
# #                  get_number([0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1])))
# # result = divide_without(test,4, 16)
# test = np.array((get_number([1,0,1,0,1,0]),
#                  get_number([0,1,0,1,0,1]),
#                  get_number([1,1,0,1,0,1]),
#                  get_number([1,1,1,1,1,1]),
#                  get_number([0,1,1,1,1,1]),
#                  get_number([0,1,0,1,0,1])))
# result = divide_without(test,1)
# bits = np.array([get_bits(n,5) for n in result[0]])
# print(bits)
# bits = np.array([get_bits(n,5) for n in result[1]])
# print(bits)

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
    data: np.array
    level: int
    code: str
    value: str
    operation: DefineBitOp | None
    entropy: EntropyDiff | None
    total_bit: int
    remaining_bits: int

    def get_commons(s):
        level = s.level + 1
        node_name = f"lvl:{level},{s.operation}"
        code = s.code + str(s.operation.bit) if s.level > 0 else ''
        value = "".join([str(s.operation.bit) if c == Char.branch
                         else c
                         for c in s.value])
        return BuildParamsCommons(level, node_name, code, value)

    def create_child(self, data, value, operation, entropy, keep_code=False):
        bit = operation.idx
        return BuildParams(data=data,
                           level=self.level + 1,
                           code=self.code if keep_code else self.code + str(bit),
                           value=value,
                           operation=operation,
                           entropy=entropy,
                           total_bit=self.total_bit,
                           remaining_bits=self.remaining_bits - 1)
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
                           remaining_bits=bit_count)
        pass


class BuildParamsCommons(NamedTuple):
    level: int
    node_name: str
    code: str
    value: np.array


class Split(NamedTuple):
    bit_idx: int
    items_with: np.ndarray
    items_wout: np.ndarray
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
        # self.mgr.print_realtime = False
        self.node, self.avg_bits, self.leaf_ext, self.bins_ext, self.codes = self._build()

    def get_avg_bits(self):
        return self.avg_bits

    def get_next_split(s, data, bit_count):
        begin_entropy_list = get_bitwise_entropy(data, bit_count)
        begin_entropy = sum(begin_entropy_list)
        max_g = 0
        max_with_parts = None
        max_wout_parts = None
        max_with_entro = []
        max_wout_entro = []
        max_bit_idx = 0
        increases = []
        for i in range(bit_count):
            with_parts, wout_parts = divide_without(data, i)
            with_e_list = get_bitwise_entropy(with_parts, bit_count - 1)
            wout_e_list = get_bitwise_entropy(wout_parts, bit_count - 1)
            with_g = begin_entropy - np.sum(with_e_list)
            wout_g = begin_entropy - np.sum(wout_e_list)
            gain = (with_g * len(with_parts) + wout_g * len(wout_parts)) / (len(with_parts) + len(wout_parts))
            if gain > max_g:
                max_with_parts = with_parts
                max_wout_parts = wout_parts
                max_with_entro = with_e_list
                max_wout_entro = wout_e_list
                max_bit_idx = i
                max_g = gain
            increases.append(gain)
        return Split(bit_idx=max_bit_idx,
                     items_with=max_with_parts,
                     items_wout=max_wout_parts,
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

        def get_single_value_bits(data, bit_count):
            result = []
            for i in range(bit_count):
                bit = np.pow(2, bit_count - i - 1)
                values = data & bit
                max = np.max(values)
                min = np.min(values)
                if max == min:
                    bit_idx = int(np.log2(bit))
                    is_bit = 1 if min > 0 else 0
                    result.append((bit_idx, is_bit))
            return result

        def check_defined(params: BuildParams, value: str | None = None):
            nonlocal leaf_len, strait_count
            value = params.value if value is None else value

            defined = get_single_value_bits(params.data, params.remaining_bits)
            if len(defined) == 0:
                return []

            result = []

            for op in [DefineBitOp(i, b) for i, b in defined]:
                value = self.merge_str(value, bit_str(op.idx, bit=op.bit, l=params.remaining_bits))
                data = get_without_bit(params.data, op.idx)
                params = params.create_child(data=data,
                                             operation=op,
                                             value=value,
                                             entropy=None,
                                             keep_code=True)

                flag_len.append(len(get_bits(op.idx)))
                strait_count += 1
                result.append(params)

            if len(get_single_value_bits(params.data, params.remaining_bits)) > 0:
                raise Exception("Not all bits are defined")

            return result

        def build_recursive(params: BuildParams):
            if params.data.size == 1:
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

            true_node = build_recursive(params.create_child(split.items_with, out_val,
                                                            operation=DefineBitOp(split.bit_idx, bit=1),
                                                            entropy=EntropyDiff(split.entropy_begin,
                                                                                split.entropy_after_with,
                                                                                split.bit_idx)))
            false_node = build_recursive(params.create_child(split.items_wout, out_val,
                                                             operation=DefineBitOp(split.bit_idx, bit=0),
                                                             entropy=EntropyDiff(split.entropy_begin,
                                                                                 split.entropy_after_wout,
                                                                                 split.bit_idx)))
            node_count += 1
            return Node(value, true_node, false_node)

        def create_leaf(params):
            nonlocal depths, leaf_len, codes, leaf_count
            data, parent_level, parent_code, parent_value, op, entropy, bit_count, remaining_bits = params
            level, node_name, code, value = params.get_commons()
            depths.append(level)

            leaf_value = data[0]
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

            params = BuildParams.make_root(data=data, bit_count=self.bit_count)

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

        def build_bins_n_print(v, q):
            print(f"Max: {np.max(v)}, Avg: {np.average(v)}")
            bins = np.bincount(v)
            print(f"Distribution: {bins}")
            value_perc = np.array(np.percentile(v, q=q), dtype=np.uint32)
            value_perc_delta = value_perc - np.concat((np.zeros(1, dtype=np.uint32), value_perc[:-1]))
            print(f"Q{q}: {value_perc}->{value_perc_delta}")
            return value_perc_delta

        print("==Data==")
        print("Codes:", len(codes), "Nodes:", node_count, "Straits:", strait_count, "Leafs:", leaf_count)
        print("==Leafs==")
        leaf_ext_delta = build_bins_n_print(leaf_len, [45, 90, 100])
        print("==Flags==")
        bins_ext_delta = build_bins_n_print(flag_len, [45, 90, 100])

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
        #         # n.bit
                pass

    def compression_ratio(self, original_bits=16):
        return self.average_bits() / original_bits

    def print(self):
        print(self.mgr)


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

    coder.print()

    print(f"{name}: avg_bits={avg_bits:.3f}, compression={ratio_bits:.3f}, {avg_bits - bits_to_take:.3f}")
    print("END")



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