# Multislice

## Zweck

Selektion **mehrerer Bit-Bereiche (Runs)** aus gepackten Wörtern (`CommonNBitAry`).
Die Indizes können lazy gesammelt und erst beim Lesen in ein `Multislice` zusammengefasst
werden — die Konstruktion (Segmentierung) ist einmalig (~40 µs bei 64 Bits) und
damit amortisiert.

## Funktionsweise

```
__init__      breaks bestimmen            np.diff != 1
              slices daraus              src_start / src_stop
              länge bestimmen            lengths

get_slices    zip über src_start / src_stop (berechnet nichts vor)

select_bits   verschiebung ermitteln      shift = tgt_lsb - src_lsb
              bitmask erstellen          (1 << lengths) - 1 << src_lsb
              anwenden                    word & mask
              verschieben                << / >> (signed shift!)
              zusammenfügen              bitwise_or.reduce
```

- Bit 0 = MSB (wie BitFlagArray), Runs landen im Ziel in Auswahl-Reihenfolge.
- Ziel-Anordnung (`total_len - cumsum(lengths)`) wird lazy in `select_bits` berechnet.
- Shifts müssen **signed** gerechnet werden (`np.intp`), sonst wrappt uint.

## Benchmarks

10.000 Wörter × 64 Bit (uint64), Best-of-5 über je 100 Aufrufe, Python 3.14 / NumPy 2.5.
Alle Varianten liefern Ergebnisgleichheit (per Assert verifiziert).

### Bit-Selektion: Multislice vs. per-Index vs. entpackte 1-Bit-Items

| Selektion | Multislice | per-Index (vektoriell) | 1-Bit-Items (entpackt) |
|---|---:|---:|---:|
| K=8,  1 Run   | **34 µs** | 466 µs  | 49 µs  |
| K=32, 1 Run   | **34 µs** | 1519 µs | 192 µs |
| K=64, 1 Run   | **33 µs** | 2693 µs | 382 µs |
| K=32, 32 Runs | 2033 µs | 1496 µs | **189 µs** |
| K=16, ~9 Runs | 1137 µs | 908 µs  | **97 µs**  |

### Item-Selektion (nativ, gepackte Wörter)

| Zugriff | Zeit |
|---|---:|
| Slice `[0:10000]` (View) | ~0 |
| Fancy, 10.000 Indizes | 6.6 µs |
| Fancy, 5.000 Indizes | 3.3 µs |

## Fazit / Faustregeln

1. **Multislice skaliert mit der Anzahl Runs, nicht mit der Bitlänge:**
   1 Run kostet ~34 µs — egal ob 8 oder 64 Bits selektiert werden.
2. **Runs (auch kurze!) sind der Gewinnfall:** Faktor 14–80 gegen per-Index,
   Faktor 1.5–11 gegen entpackte Items. Der Umschlagpunkt liegt bei einer
   mittleren Run-Länge von ~3–4 Bits, **nicht** bei der absoluten Bitlänge.
3. **Stark fragmentierte Selektionen** (viele Singletons) sind der einzige
   Verlustfall: per-Index und das entpackte Layout gewinnen dort (max. Faktor 1,3
   bzw. ~10). Das entpackte Layout ist fragmentierungsunabhängig, kostet aber
   8× Speicher und liefert ungepackte Bits — Repacken wäre extra.
4. **Relation zur Item-Dimension:** Eine native Item-Fancy-Selektion über das
   ganze Array kostet ~6.6 µs — eine Multislice-Bitselektion (1 Run) das
   ~5-fache. Die Bit-Dimension ist damit keine Größenordnung teurer als die
   Item-Dimension.
5. Mehrere Slices kann numpy weder auf Bit- noch auf Item-Ebene nativ — auf der
   Item-Ebene bleibt nur Fancy-Indexing (obige ~6.6 µs pro ganzem Array), auf
   der Bit-Ebene übernimmt das Multislice.
