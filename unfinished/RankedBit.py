def calc_req_counter_bits(item_count: int, defined: int, max=128) -> Tuple[int, int]:
    for bit_count in range(defined+1,max):
        possibilities = np.prod([bit_count - i
                                 for i in range(defined)])
        if possibilities >= item_count:
            return bit_count, defined
    raise ValueError("max bits exceeded")

def bits_by_rank(bit_count)-> List[List[int]]:
    flags = np.array(1 << np.arange(bit_count), dtype=np.uint8)
    return [ [sum(p) for p in combinations(flags, r + 1)]
                for r in range(0, bit_count) ]

def bits_rank_first(bit_count, min_r=0, max_r=None)->List[int]:
    max_r = bit_count if max_r is None else max_r
    flags = np.array(1 << np.arange(bit_count), dtype=np.uint8)
    # if by_rank:
    #     return { r: [sum(p) for p in combinations(flags, r + 1)]
    #                 for r in range(min_r, max_r) }
    # else:

def bits_rank_first_from_flags(bit_flags, min_r=0, max_r=None) -> np.ndarray:
    return get_as_unsigned([sum(p)
                            for r in range(min_r, max_r)
                            for p in combinations(bit_flags, r + 1)],
                           fit=True)

def get_comb_idx(comb_items, value_mask, value_rank) -> int:
    for i, c in enumerate(combinations(comb_items, value_rank + 1)):
        if sum(c) == value_mask:
            return c
    return -1

class RankedBit(NamedTuple):
    bit_mask: int
    bit_value: int
    bit_count: int

    def expand(self) -> Tuple[int,npt.ArrayLike,int, npt.ArrayLike]:
        mask_bit: np.ndarray = get_bit_flags(self.bit_mask)
        mask_rank = mask_bit.size
        value_bit: np.ndarray = get_bit_flags(self.bit_value)
        rank = value_bit.size
        return mask_rank, mask_bit, rank, value_bit

    @staticmethod
    def rank_states(mask_rank: int, val_rank: int) -> int: #Tuple[int,List[int]]:
        comb_by_rank = [comb(mask_rank, idx_r+1) for idx_r in range(val_rank)]
        return sum(comb_by_rank) #, comb_by_rank

    def value(self) -> npt.ArrayLike:
        mask_rank, mask_bit, val_rank, value_bit = self.expand()
        rank_floor = self.rank_states(mask_rank, val_rank-1)
        value_idx = get_comb_idx(mask_bit, self.bit_value, val_rank)
        scalar_value = rank_floor+value_idx
        return scalar_value

    @staticmethod
    def _build_int(value: int | npt.ArrayLike) -> int:
        if isinstance(value, int):
            return value
        else:
            ipt = np.bitwise_or.reduce(value)
            if not np.sum(value) == ipt:
                raise ValueError("val and mask mut not contain same flag twice")
            return ipt

    @classmethod
    def empty(cls, bit_count: int) -> 'RankedBit':
        return RankedBit(bit_mask=(1 << bit_count)-1, value=0, bit_count=bit_count)

    @classmethod
    def from_value(cls, val: int | npt.ArrayLike, mask: int | npt.ArrayLike = None) -> 'RankedBit':
        v = cls._build_int(val)
        m = cls._build_int(mask)
        b_cnt = m.bit_count()
        return RankedBit(bit_mask=m, value=v,  bit_count=b_cnt)


    @classmethod
    def from_bits(cls, bits: npt.ArrayLike, indices: npt.ArrayLike = None) -> 'RankedBit':
        b = get_as_unsigned(bits,fit=True)
        i = get_as_unsigned(indices,fit=True) if indices is not None else np.arange(b.size)
        t = get_type_for_bit_count(np.max(i))
        bit_count, mask, value = normalize_flags(i, b)
        return RankedBit(mask, value, bit_count)

    def __repr__(self):
        mask_rank, mask_bit, rank, value_bit = self.expand()
        mask_nrs = [f"{i}-{n}" if n >= 0 else "_" for i, n in np.ndenumerate(mask_bit)]
        mask = ", ".join(mask_nrs)
        return f"[{rank}][{mask}][{value_bit}]"


testBit = RankedBit.from_bits([1,0,1],[1,2,3])
print(testBit)

testBit = RankedBit.from_bits([1,0,1],[0,1,2])
print(testBit)


class BitGroupWalker:
    bit_groups: Dict[int, List[int]] = {}


def get_spread_set(item_count: int, n_defined: int, min_dist=2, max=128):
    # calc_req_counter_bits(item_count, defined=4)
    set_bit_count, n_defined = calc_req_counter_bits(item_count, defined=n_defined)
    set_bit_count += ((-set_bit_count) % n_defined) # round up
    bit_list = np.array(range(set_bit_count), dtype=int)
    # counter_per = np.array(list(permutations(bit_list, n_defined)))
    # spread_bits = get_bits(np.sum(1 << counter_per, axis=1))
    # zero overlap groups
    last_group=-1
    groups = Bitty.empty((item_count,set_bit_count))
    group_count = n_defined
    group_bit_count = set_bit_count // group_count
    group_bits = np.split(bit_list, group_count)
    def get_group_bit_slice(group_nr: int):
        return slice(group_nr * group_bit_count, (group_nr + 1) * group_bit_count)
    # sl = [get_group_bit_slice(g) for g in range(group_count)]
    # def get_bit_group(g: int):
    #     return groups.b[get_group_bit_slice(g)]

    g_bit_expanded = range(group_bit_count)

    ranked_bits = bits_by_rank(group_bit_count)

    index = np.array([0,0,0])
    def g_bit_list(start_r, stop_r):
        return bits_rank_first(group_bit_count, start_r, stop_r)
    def iterate_groupwise_bits(max_overlap: int, idx_r: int):
        bit_lists = [ranked_bits[idx_r] for _ in range(group_count)]
        for groupwise_bits in product(*bit_lists):
            yield groupwise_bits

    groupwise_b = [get_as_signed(i,fit=True) for i in iterate_groupwise_bits(0,0)]
    print(groupwise_b)

    bits = np.arange(12)

    group_asc = np.arange(group_count)
    # [[g,g,g,g] for g in g_bit_exp]
    s_bit_exp = [0, 1, -1]
    first = [(group_asc * s + g) % group_count
             for s in s_bit_exp for g in group_asc]

    print(np.array(first))


    print("spread_bits:")
    # print(spread_bits)
