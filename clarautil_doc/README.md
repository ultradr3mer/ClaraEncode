# BitFlagArray

A Python library containing two independent pillars:

| Pillar            | Modules                | What it provides                              |
|-------------------|------------------------|-----------------------------------------------|
| **Typed tables**   | `GenricTable`, `QueryableTable` | Row-typed structured arrays + constraint-column querying |
| **Bit storage**    | `BitFlagArray`         | Pack/unpack individual bits inside integers   |

`GenricTable` and `QueryableTable` build on each other (`QueryableTable`
extends `GenricTable`). `BitFlagArray` is independent — it provides
bit-level storage and helpers. A combination is possible but not wired up.

```
GenricTable        ← typed tables over structured arrays
    │
    └── QueryableTable   ← constraint columns + selections

BitFlagArray        ← packed bit-level storage (independent)
```

---

## BitFlagArray — packed bit storage

### What it does

NumPy stores integers as a single block of bits. `BitFlagArray` lets you
treat each integer as a **row of flag bits** and slice, mask, and
reconstruct at the bit level — without manual `<<` / `&` / `>>` arithmetic.

### Plain NumPy vs BitFlagArray

**Goal: extract bits 2..5 from every element of an 8-bit array.**

Plain NumPy:

```python
import numpy as np

arr = np.array([0b10110010, 0b11110000, 0b00001111], dtype=np.uint8)

# Manual shift-and-mask for bits [2, 6)
mask   = (1 << (6 - 2)) - 1        # 0b00111100 ... wait, 0b00001111
shifted = arr >> 2
result  = shifted & mask           # array([ 12,  12,   3])
# Which bit positions were these? You have to track the shift by hand.
```

BitFlagArray:

```python
from BitFlagArray import NBitAryOnly

val = NBitAryOnly(np.array([0b10110010, 0b11110000, 0b00001111], dtype=np.uint8), bit_count=8)

result = val[2:6]   # bits 2,3,4,5 → same values, self-documenting
```

The slice `[2:6]` reads exactly like array slicing — because it **is**
array slicing, just at the bit granularity.

### Use cases

- **Flag sets**: store 64 boolean flags in one `uint64` and query
  individual flags by position.
- **Bitmasks**: compose/decompose permission bits, status bits, feature
  flags.
- **Compact encodings**: pack multiple small fields into one integer
  column.
- **Type detection helpers**: `common.py` provides `get_type_for_bit_count`
  and `get_type_for_scalar` which use the lookup-table pattern to pick the
  smallest NumPy dtype for a given value or bit count — the same pattern
  that `QueryableTable` formalizes.

### Key API

```python
from BitFlagArray import NBitAryOnly

val = NBitAryOnly(data, bit_count=8)

val[2:6]           # slice bits 2..5
val[0]             # single bit
val[2:6] = 0b1010  # assign bits
len(val)           # number of bits
```

---

## GenricTable — typed tables over structured arrays

### What it does

A NumPy structured array has named fields, but you must specify the dtype
by hand and field access is via `arr['name']` strings. `GenricTable`
lets you **define a row** (a `NamedTuple`) and derives the dtype, the
columns, and the row iteration automatically.

### Plain NumPy vs GenricTable

**Goal: store a table of (signed, max, bits) rows and access columns.**

Plain NumPy:

```python
import numpy as np

# Must write the dtype by hand
dt  = np.dtype([('signed', np.bool), ('max', np.uint64), ('bits', np.uint8)])
arr = np.array([(False, 255, 8), (False, 65535, 16), (True, 127, 8)], dtype=dt)

arr['signed']   # array([False, False, True])
arr['max']      # array([255, 65535, 127])
arr[0]          # (False, 255, 8)   ← a numpy.void, no attribute access
```

GenricTable:

```python
from typing import NamedTuple
from GenericTable import PlainTable


class TypeRow(NamedTuple):
  signed: np.bool
  max: np.uint64
  bits: np.uint8


class TypeTable(PlainTable[TypeRow]):
  @classmethod
  def get_row_type(cls) -> type[TypeRow]:
    return TypeRow


tbl = TypeTable.build("types", [
  (False, 255, 8),
  (False, 65535, 16),
  (True, 127, 8),
])

tbl.signed  # array([False, False, True])       ← attribute, not string
tbl.max  # array([255, 65535, 127])
tbl[0]  # TypeRow(signed=False, max=255, bits=8)  ← typed NamedTuple
```

### What goes in

`build(name, data)` accepts anything `np.asarray` can convert:

```python
# List of tuples
TypeTable.build("t", [(False, 255, 8), (True, 127, 8)])

# List of lists
TypeTable.build("t", [[False, 255, 8], [True, 127, 8]])

# Existing structured array
TypeTable.build("t", existing_structured_array)
```

The dtype is derived from the row annotations:

```python
TypeRow.__annotations__
# {'signed': numpy.bool, 'max': numpy.uint64, 'bits': numpy.uint8}

TypeTable.dtype
# dtype([('signed', '?'), ('max', '<u8'), ('bits', 'u1')])
```

### How you access it

| Access            | Plain NumPy       | GenricTable                  |
|-------------------|-------------------|------------------------------|
| Column            | `arr['max']`      | `tbl.max` (attribute)        |
| Single row        | `arr[0]` (void)   | `tbl[0]` (NamedTuple)         |
| Iterate rows      | `for r in arr:` (void) | `for r in tbl:` (NamedTuple) |
| All rows          | —                 | `tbl.rows()` → list           |
| Length            | `len(arr)`        | `len(tbl)`                    |
| Dtype             | `arr.dtype`       | `tbl.dtype`                   |
| Table name        | —                 | `tbl.name`                    |

```python
for row in tbl:
    print(row.signed, row.max, row.bits)
# False 255 8
# False 65535 16
# True 127 8
```

### The class hierarchy

```
Table[TRow, TCreator]  (ABC)
│   ├── build(name, data)         ← creator classmethod
│   ├── __iter__                  ← yields typed rows
│   ├── __getitem__(int)          ← returns a row NamedTuple
│   ├── __len__, dtype, rows()
│   └── _col_creator_cls          ← set by intermediate bases
│
├── PlainTable[TRow]              ← columns = np.ndarray
│       _col_creator_cls = NPContainerCreator
│
└── QueryableTable[TRow]          (in QueryableTable.py)
        _col_creator_cls = ConstraintColCreator
```

### The column-adapter pattern

Column construction is formalized through a creator/adapter pair so that
adding a new column type only requires two small classes:

```
TColContainerCreator  (abstract factory)
    └── get_adapter(parent, name) → TColContainerAdapter

TColContainerAdapter  (abstract adapter)
    └── init_column(ary, cell_type) → column
```

| Creator                | Adapter               | Column type        |
|------------------------|-----------------------|--------------------|
| `NPContainerCreator`  | `NPColAdapter`        | `np.ndarray`       |
| `ConstraintColCreator` | `ConstraintColAdapter` | `ConstraintColumn` |

### TableCreator

`TableCreator` binds a table type and its row type so the return type
is known to type checkers:

```python
from GenericTable import TableCreator

creator = TableCreator[TypeRow, TypeTable](TypeTable)
tbl = creator.build("types", data)  # type: TypeTable
```

### Use cases

- **Lookup tables**: dtype catalogs, enum mappings, config tables.
- **Schema-driven data**: the row type is the schema; the table is the
  data — changes to the row type automatically update the dtype.
- **Any structured NumPy data**: when you want attribute access and
  typed rows instead of `arr['field']` and `numpy.void`.

---

## QueryableTable — constraint columns and selection pipelines

### What it does

A `QueryableTable` is a `GenricTable` where each column is a
**`ConstraintColumn`** — a wrapper that supports comparison operators
(`==`, `>=`, `<`, ...) and returns a **`ConstraintSelection`** (a set
of matching row indices) instead of a boolean array.

Selections can be chained with `and` or `&` to build multi-condition
queries that read like SQL `WHERE` clauses.

### Plain NumPy vs QueryableTable

**Goal: find the smallest unsigned type whose max >= 123.**

Plain NumPy:

```python
arr = np.array([(False,255,8,-1),(False,65535,16,255),(True,127,8,-1),...], dtype=dt)

# Manual: filter, then filter again, track indices by hand
unsigned = arr[arr['signed'] == False]
big_enough = unsigned[unsigned['max'] >= 123]
# Now you've lost the original row positions and have a copy.
```

QueryableTable:

```python
from QueryableTable import QueryableTable

class TypeLookup(QueryableTable[TypeRow]):
    @classmethod
    def get_row_type(cls) -> type[TypeRow]:
        return TypeRow

tbl = TypeLookup.build("lookup", data)

smallest = (
    tbl.signed == False      # ConstraintSelection([0,1,2,3])
    and tbl.max >= 123       # refine → [0,1,2]
    and tbl.prev_max < 123   # refine → [0]
)
# ConstraintSelection(indices=array([0]))
```

### Selection pipelines

Two equivalent forms:

**`and`-form** (idiomatic, no parens — `and` binds looser than comparisons):

```python
sel = (
    tbl.signed == False
    and tbl.max >= 123
    and tbl.prev_max < 123
)
```

**`&`-form** (explicit, parens required — `&` binds tighter than comparisons):

```python
sel = (
    (((tbl.signed == False) & tbl.max) >= 123) & tbl.prev_max
) < 123
```

### How the pipeline works

```
signed == False       →  ConstraintSelection([0,1,2,3])   # select
& max                 →  restrict `max` to rows [0,1,2,3]  # restrict
>= 123                →  ConstraintSelection([0,1,2])       # refine
& prev_max            →  restrict `prev_max` to [0,1,2]     # restrict
< 123                 →  ConstraintSelection([0])          # refine
```

Internally:

1. `column op value` → `ConstraintSelection` with **global** row indices.
2. `selection & column` → a **restricted column** (the column's data
   sliced to `selection.indices`), which remembers `parent_indices`.
3. The next comparison on the restricted column maps its local mask
   back to global indices (`parent_indices[local_mask]`).

### The `and`-trick

Python's `and` does **not** call `__and__`. It evaluates `bool(left)` and
returns the right side if truthy. `ConstraintSelection.__bool__` pushes
`self` onto a pending stack and returns `True`; the next column
comparison pops it and applies it as a restriction.

**Limitations:**

- `is` is not overloadable — use `== Undefined` for "any value".
- The right side of `and` must be a comparison. A bare
  `selection and column` (no `>=`/`<`/...) leaves a dangling stack entry.
- Not thread-safe (module-global pending stack).

### ConstraintColumn operators

| Operator | Returns                    | Meaning            |
|----------|----------------------------|--------------------|
| `==`     | `ConstraintSelection`      | equal              |
| `!=`     | `ConstraintSelection`      | not equal          |
| `<`      | `ConstraintSelection`      | less than          |
| `<=`     | `ConstraintSelection`      | less or equal      |
| `>`      | `ConstraintSelection`      | greater than       |
| `>=`     | `ConstraintSelection`      | greater or equal   |

### ConstraintSelection operators

| Operator | Meaning                                  |
|----------|------------------------------------------|
| `&`      | selection ∩ selection (intersection)    |
| `&`      | selection & column (restriction)         |
| `and`    | same as `&` (via the trick)               |

### Using a selection

```python
sel = tbl.signed == False and tbl.max >= 1000

sel.indices            # array of matching row positions
tbl.max[sel.indices]   # the matching max values
tbl[sel.indices[0]]    # first matching row (TypeRow)
```

### Filtering rows via iteration

```python
signed_rows = [row for row in tbl if row.signed == True]
# [TypeRow(signed=True, max=127, ...), ...]
```

### Use cases

- **Type lookup**: "smallest unsigned type that can hold value X" — the
  example above.
- **Range queries**: `price >= 100 and price <= 500`.
- **Multi-criteria filtering**: any combination of column comparisons
  chained with `and`.
- **Any structured-data query** where you want index-based selections
  instead of boolean masking + index recovery.

---

## Module reference

### `GenericTable.py`

| Name                        | Kind        | Description                              |
|-----------------------------|-------------|------------------------------------------|
| `NpColDef`                  | NamedTuple  | `(name, type)` column definition         |
| `FieldSpec`                 | type alias  | `list[tuple[str, DTypeLike]]`            |
| `to_col_defs(spec)`         | function    | Field-spec → `list[NpColDef]`            |
| `TColContainerAdapter`      | ABC         | Adapter: `init_column(ary, cell_type)`   |
| `TColContainerCreator`      | ABC         | Factory: `get_adapter(parent, name)`     |
| `NPColAdapter`              | class       | Adapter for `np.ndarray` columns         |
| `NPContainerCreator`        | class       | Factory for numpy columns                |
| `Table[TRow, TCreator]`     | ABC         | Abstract table base                      |
| `PlainTable[TRow]`          | class       | Table with `np.ndarray` columns          |
| `TableCreator[TRow, TTable]` | class      | Generic table factory                    |

### `QueryableTable.py`

| Name                      | Kind      | Description                              |
|---------------------------|-----------|------------------------------------------|
| `ConstraintSelection`     | dataclass | `indices` + `&` + `and`-trick           |
| `ConstraintColumn`        | class     | Column with comparison → selection        |
| `ConstraintColAdapter`    | class     | Adapter for `ConstraintColumn`           |
| `ConstraintColCreator`    | class     | Factory for constraint columns            |
| `QueryableTable[TRow]`    | class     | Table with `ConstraintColumn` columns     |

---

## Running the tests

```bash
pytest Test/test_GenricTable.py Test/test_QueryableTable.py -v
```

32 tests cover: column definitions, table construction, iteration,
indexing, all comparison operators, both pipeline forms (`and` and `&`),
selection intersection, and the `TableCreator` factory.
