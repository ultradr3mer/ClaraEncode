# AGENTS

## Projekt
BitFlagArray — gepackte Bit-Daten (BitFlagArray/Bitty), typisierte Tabellen über numpy-Strukturarrays, Bit-Selektion.

## Kommunikation
- Der Nutzer schreibt Deutsch — auf Deutsch antworten, knapp und direkt.
- Docstrings/Kommentare minimal halten — lange Docstrings löscht der Nutzer.

## Packaging
- Die 7 Kern-Module liegen im Paket **`clarautils/`** (imports: `from clarautils import Bitty`, `from clarautils.commonTyping import ...`). Interne Imports im Paket sind **relativ** (`from .common import ...`).
- Installierbar per `pyproject.toml` (setuptools, `numpy` als einzige Abhängigkeit): `pip install -e .` im Ziel-venv — Änderungen hier wirken sofort.
- `clarautils/__init__.py` re-exportiert die Kern-API — neue öffentliche Namen dort UND in `__all__` ergänzen.
- Repo-Rest (showcase, DebugPrint, association, reverseEncoding, GainCoder, Tests) importiert `clarautils.<Modul>` absolut.

## Modul-Landkarte
| Modul | Inhalt |
|---|---|
| `clarautils/common.py` | `get_first_or`, `ExceptionRaiser`/`ExcRaiser`, `iter_bits` — **keine** `get_type_*` mehr |
| `clarautils/commonTyping.py` | `INTEGER_TYPES`, `get_type_for_scalar/array/bit_count`, `get_as_signed/unsigned`, `DTableFields`, `ExcRaiser`-Instanzen |
| `clarautils/GenericTable.py` | Table-Framework (`TableFields`, item/range-Varianten, `Table`) — hieß früher `GenricTable.py` |
| `clarautils/QueryableTable.py` | lazy `Query`/`Constraint`, `ConstraintColumn`=`CCol`, `QueryableTable`, `QTblSpecialCol`, `Undefined` |
| `clarautils/commonEncoding.py` | `CommonNBitAry`/`CommonNBitSc`, `get_number`/`get_bits` |
| `clarautils/BitFlagArray.py` | `BitFlagArray`/`Bitty`, `SliceView`, LRU-Cache |
| `clarautils/Mulitslice.py` | `Multislice` (Schreibweise "Mulitslice" ist Legacy — **nicht umbenennen**) |
| `Multislice.md` | Benchmarks + Faustregeln zur Bit-Selektion |
| `GenericTableOld.py` | deprecated, Importblock auskommentiert — unberührt lassen |
| `README.md` | **veraltet** (PlainTable/TableCreator-Ära) — nicht vertrauen |

## Kernkonzepte / Terminologie (vom Nutzer festgelegt)
- "Row type" heißt jetzt **table type**; er zerfällt in **Range** (Spalten-Container, z. B. `CCol`) und **Item** (Skalar, z. B. `np.bool_`). Generischer Parameter: `TRowNRange` ("Row aNd Range").
- Felder werden **genau einmal** deklariert — als `Range | Item`-Union auf einer `TableFields`-Unterklasse. Die Deklaration IST der table type:
  ```python
  class DTableFields(TableFields):
      signed: CCol | np.bool_
  class TypeTable(QTblSpecialCol[DTableFields, Type[Any]], DTableFields): ...
  ```
  Das Framework generiert intern `item_type` (skalar-getyptes NamedTuple, gecacht pro Deklaration) und `range_type` (z. B. `DTableRange`). Es gibt KEIN `table_type_from_fields` mehr — nichts per Hand ableiten.
- `tbl[int]` → Item-Variante (Zeile); `tbl[slice]`/`tbl[indices]` → Range-Variante (Spalten über die Selektion).
- Spaltentypen bestimmt die Deklaration (Validierung beim Bau → `TypeError` bei Mismatch). Familien liefern nur `build_column` + `default_range`.
- Query-Modell ist **lazy**: Vergleiche auf Spalten bauen `Constraint`-Blätter, `&`/`|` komponieren zu `Query`; Auswertung erst bei `.indices`/`get_first`/`get_all`. Der alte `and`-Trick/`_pending_selections` ist entfernt — nicht wieder einführen.
- `Undefined = Literal` (QueryableTable): `column == Undefined` → immer-wahre Bedingung (bei `get_type_for_*` = "beide Vorzeichen-Familien").
- `_find_table_fields` (MRO-Scan) muss Framework-Klassen überspringen — Tabellen erben die Deklaration und matchen sonst selbst.
- Alte Namen (`DRow`, `row_type`, `ConstraintSelection`, Adapter/Creator-Klassen, `GenricTable`) existieren nicht mehr.

## commonTyping-Regeln
- `signed`-Tri-State: `False` (Legacy-Default, nur unsigned), `True` (nur signed), `Undefined` (beide Familien).
- `_get_type_for_bounds`: `abs_min` grenzt die negative Seite ab (-128 passt in int8!); `low=0` deaktiviert das (Legacy-unsigned-Pfad prüft nur `max`).
- Fehlermeldungen **verbatim** (Tippfehler sind Absicht): `Exception("value to big")` / `Exception("to many bits requested")`.
- Annotation `signed: "bool | Undefined"` muss ein **String** sein — `bool | Literal` crasht zur Laufzeit.
- Funktionen, die `INTEGER_TYPES` mehrfach nutzen: lokal `type_tbl = INTEGER_TYPES` aliasen.

## Mulitslice / Bit-Selektion
- Konvention MSB-first (Bit 0 = MSB), wie BitFlagArray; Runs landen im Ziel in Auswahl-Reihenfolge.
- Shifts **immer signed** rechnen (`.astype(np.intp)`) — unsigned wrappt (-4 → 252).
- Attributnamen: `src_start`/`src_stop` (nicht `start`/`stop`).
- `get_type_for_bit_count(np.max(...))` ist ein Wert-als-Bitcount-Bug — für Werte `get_type_for_scalar` nutzen.
- Multislice skaliert mit Run-Anzahl, nicht mit Bitlänge; Umschlagpunkt zu per-Index bei mittlerer Run-Länge ~3–4 Bits (Details: `Multislice.md`).

## Tests & Aufruf
- Windows/pwsh, Python 3.14 in `.venv`:
  `& .\.venv\Scripts\python.exe -m pytest Test -q`
- **Bewusst defekt** (damalige Scope-Entscheidung, bei Bedarf mit dem Nutzer abklären): `Test/test_common.py`, `Test/test_DebugPrint.py` — stale `from common import get_type_for_*`.
- `Test/test_perf*.py` sind Benchmarks (~50 s); `Test/conftest.py` hat eine autouse-Fixture, die BitFlagArray importiert.
- Referenz ohne die beiden defekten + Perf: 209 passed.
- Schnellster Smoke-Test: die `__main__`-Demos der Module laufen lassen: `& .\.venv\Scripts\python.exe -m clarautils.GenericTable` (analog `clarautils.QueryableTable`, `clarautils.commonTyping`, `clarautils.Mulitslice`).

## Workflow
- **Nicht** committen/pushen — der Nutzer committet selbst ("update"-Nachrichten).
- Vorsicht bei `git stash`: zwei Stashes des Nutzers vorhanden (einer wurde in einer Session versehentlich gedroppt und wiederhergestellt — stash-Operationen doppelt prüfen).
- `rg` ist nicht installiert — grep-Tool/`Select-String` nutzen.
- Benchmarks für Multislice liegen außerhalb des Repos (Temp), reproduzierbar per `Multislice.md`-Zahlen.
