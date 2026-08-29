from enum import StrEnum
from typing import Any, NamedTuple

import numpy as np

from clarastrings import FramePrint
from clarastrings import BeginItemOptions, ParentChildRelation, ItemClosingBeavior
from clarautils import symbol_to_str


class Char(StrEnum):
    fill = "."
    down = "↧"
    branch = "X"
    space = " "


def get_diff(string_a, sting_b):
    return "".join([Char.down if a != b
                    else Char.space
                    for a, b in zip(string_a, sting_b)])


def entropy_base_10_str(val):
    ceiling = np.array(np.ceil(val * 9), dtype=np.int32)
    ceiling = np.min((ceiling, np.full_like(ceiling, fill_value=9)), axis=0)
    return symbol_to_str(ceiling)


def entropy_before_str(entropy):
    return entropy_base_10_str(entropy.entropy_before)


def entropy_after_str(entropy):
    result = entropy_base_10_str(entropy.entropy_after)
    return "".join([(Char.space if i == entropy.idx else '') + c for i, c in enumerate(result)])


def entropy_gain_sum(entropy):
    return sum(entropy.entropy_before) - sum(entropy.entropy_after)


class RootBegin(NamedTuple):
    value: str


class NodeBegin(NamedTuple):
    name: str
    value_in: str
    value: str
    entropy: Any


class Strait(NamedTuple):
    op: Any
    value_before: str
    value_after: str


class NodeSplit(NamedTuple):
    bit_idx: int
    value_before: str
    value_after: str


class RootSplit(NamedTuple):
    value_before: str
    value_after: str
    gain: Any


class NodeEnd(NamedTuple):
    name: str


class Leaf(NamedTuple):
    name: str
    parent_value: str
    value: str
    full: str
    leaf_value: Any
    leaf_str: str


class TreePrinter:
    def __init__(self, realtime=False):
        self.mgr = FramePrint().get_mgr()
        self.mgr.print_realtime = realtime
        self.sb = None
        self.offset = 0
        self._handlers = {
            RootBegin: self.on_root_begin,
            NodeBegin: self.on_node_begin,
            Strait: self.on_strait,
            NodeSplit: self.on_node_split,
            RootSplit: self.on_root_split,
            NodeEnd: self.on_node_end,
            Leaf: self.on_leaf,
        }

    def handle(self, event):
        self._handlers[type(event)](event)

    def print(self):
        print(self.mgr)

    def on_root_begin(self, e):
        sb = self.mgr.begin_item("root", options=BeginItemOptions(parent=ParentChildRelation.DirectParentIsParent))
        sb = sb.append("(root)     [")
        self.offset = sb.get_cursor()
        self.sb = sb.append(e.value).make_next_line()

    def on_node_begin(self, e):
        sb = self.mgr.begin_item(e.name, options=BeginItemOptions(parent=ParentChildRelation.DirectParentIsParent))
        sb = sb.append(f"({e.name})[in: ")
        offset = sb.get_cursor()
        sb.a(f"{e.value_in},→ entropy [in: ")
        offset2 = sb.get_cursor()
        before = entropy_before_str(e.entropy)
        after = entropy_after_str(e.entropy)
        sb = sb.a(f"{before},").make_next_line() \
            .fill_to(end="changes: ", to=offset).a(f"{get_diff(e.value_in, e.value)},") \
            .fill_to(to=offset2).a(
            f"{get_diff(before, after)},").make_next_line() \
            .fill_to(end="node: ", to=offset).a(f"{e.value},→") \
            .fill_to(end="etp node: ", to=offset2).a(
            f"{after}] gain:{entropy_gain_sum(e.entropy):.3f}").make_next_line()
        self.offset = offset
        self.sb = sb

    def on_strait(self, e):
        self.sb = self.sb.fill_to(end=f"{e.op}: ", to=self.offset).a(
            f"{get_diff(e.value_before, e.value_after)},").make_next_line() \
            .fill_to(end="now: ", to=self.offset).a(f"{e.value_after},").make_next_line()

    def on_node_split(self, e):
        self.sb = self.sb.fill_to(end=f"op:{e.bit_idx}={Char.branch}: ", to=self.offset).a(
            f"{get_diff(e.value_before, e.value_after)},").make_next_line() \
            .fill_to(end="out: ", to=self.offset).a(f"{e.value_after}]").make_next_line()

    def on_root_split(self, e):
        self.sb = self.sb.fill_to(end="changes: ", to=self.offset).a(
            f"{get_diff(e.value_before, e.value_after)}, gain:{e.gain}").make_next_line() \
            .fill_to(end="out: ", to=self.offset).a(f"{e.value_after}]")

    def on_node_end(self, e):
        self.sb = self.mgr.close_item(e.name)
        self.sb.append(f"({e.name}) End")

    def on_leaf(self, e):
        sb = self.mgr.begin_item(e.leaf_value,
                                 options=BeginItemOptions(parent=ParentChildRelation.DirectParentIsParent,
                                                          closing_beavior=ItemClosingBeavior.NoClosingTagFitChildren))
        sb = sb.append(f"({e.name})[in: ")
        offset = sb.get_cursor()
        sb.a(f"{e.parent_value}, leaf: ")
        offset2 = sb.get_cursor() + 8
        sb.a(f"({e.leaf_value})-[{e.leaf_str}]").make_next_line() \
            .fill_to(end="changes: ", to=offset).a(f"{get_diff(e.parent_value, e.value)},") \
            .fill_to(end="╰→[node: ", to=offset2).a(f"{e.value},").make_next_line() \
            .fill_to(end="node: ", to=offset).a(f"{e.value}]") \
            .fill_to(end="changes: ", to=offset2).a(f"{get_diff(e.value, e.full)},").make_next_line() \
            .fill_to(end="out: ", to=offset2).a(f"{e.full}]").make_next_line()
        self.sb = sb
