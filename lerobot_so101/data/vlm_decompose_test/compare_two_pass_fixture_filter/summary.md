# Decomposition: two-pass vs two-pass + fixture filter

- two-pass: `vlm_decompose_test/2026-10-02_07-42-16/results.json`
- two-pass + fixture filter: `vlm_decompose_test/2026-10-02_11-35-25/results.json`
- 195 scene x prompt pairs present in both runs, one trial each

## Checks

| check | two-pass | two-pass + fixture filter | change | p (exact McNemar) |
|---|---|---|---|---|
| all three objects identified | 0.73 | 0.73 | +0.00 | 1 |
| right objects planned | 0.71 | 0.72 | +0.01 | 0.62 |
| order respected | 0.95 | 0.95 | -0.01 | 1 |
| right number of sub-tasks | 0.62 | 0.69 | +0.07 | 0.0044 |
| destination is the tray | 0.98 | 1.00 | +0.02 | 0.25 |
| overall correct | 0.56 | 0.62 | +0.06 | 0.013 |

## Overall correct by category

| category | n | two-pass | two-pass + fixture filter | change |
|---|---|---|---|---|
| base | 39 | 0.46 | 0.64 | +0.18 |
| omission | 39 | 0.44 | 0.51 | +0.08 |
| sequencing | 39 | 0.67 | 0.69 | +0.03 |
| conditional | 39 | 0.56 | 0.56 | +0.00 |
| quantity | 39 | 0.69 | 0.69 | +0.00 |

## Case by case (same pairs)

Exact McNemar test on the 17 pairs where the two disagree: p = 0.013.

| outcome | pairs |
|---|---|
| both correct | 107 |
| only two-pass + fixture filter correct | 14 |
| only two-pass correct | 3 |
| both wrong | 71 |

## Failure modes (wrong cases)

| failure mode | two-pass | two-pass + fixture filter |
|---|---|---|
| missed_object | 42 | 53 |
| planned_fixture | 27 | 0 |
| wrong_count | 2 | 5 |
| wrong_objects | 14 | 16 |

## Objects missed by the perception step (cases)

| object | two-pass | two-pass + fixture filter |
|---|---|---|
| banana | 0 | 0 |
| toy | 0 | 0 |
| pouch | 53 | 53 |

## Overall correct by prompt

| category | prompt | n | two-pass | two-pass + fixture filter | change |
|---|---|---|---|---|---|
| base | put everything on the tray | 39 | 0.46 | 0.64 | +0.18 |
| omission | put everything on the tray except the banana | 7 | 0.14 | 0.29 | +0.14 |
| omission | put everything on the tray except the toy | 7 | 0.57 | 0.57 | +0.00 |
| omission | put everything on the tray except the pouch | 7 | 0.43 | 0.29 | -0.14 |
| omission | put everything on the tray except the banana and the toy | 6 | 0.83 | 1.00 | +0.17 |
| omission | put everything on the tray but leave the toy and the pouch | 6 | 0.50 | 0.67 | +0.17 |
| omission | put everything on the tray, don't touch the pouch | 6 | 0.17 | 0.33 | +0.17 |
| sequencing | put the banana on the tray first, then the toy, then the pouch | 8 | 0.62 | 0.62 | +0.00 |
| sequencing | put the pouch on the tray first, then the banana, then the toy | 8 | 0.88 | 0.88 | +0.00 |
| sequencing | put everything on the tray, starting with the toy | 8 | 0.62 | 0.62 | +0.00 |
| sequencing | put everything on the tray, but the pouch goes last | 8 | 0.50 | 0.50 | +0.00 |
| sequencing | put everything on the tray, put the banana after the toy | 7 | 0.71 | 0.86 | +0.14 |
| conditional | put the fruit on the tray | 7 | 0.43 | 0.43 | +0.00 |
| conditional | put the brown objects on the tray | 7 | 0.57 | 0.57 | +0.00 |
| conditional | put everything that is not edible on the tray | 7 | 0.57 | 0.57 | +0.00 |
| conditional | put only the soft, furry things on the tray | 6 | 1.00 | 1.00 | +0.00 |
| conditional | if there is a banana on the table, put only the banana on the tray; otherwise put everything on the tray | 6 | 0.00 | 0.00 | +0.00 |
| conditional | if there is an apple on the table, put the apple on the tray; otherwise put the toy on the tray | 6 | 0.83 | 0.83 | +0.00 |
| quantity | put one object on the tray | 8 | 0.88 | 0.88 | +0.00 |
| quantity | put two objects on the tray | 8 | 0.88 | 0.88 | +0.00 |
| quantity | put all three objects on the tray | 8 | 0.75 | 0.75 | +0.00 |
| quantity | put one of the brown objects on the tray | 8 | 0.12 | 0.12 | +0.00 |
| quantity | put two objects on the tray, but not the banana | 7 | 0.86 | 0.86 | +0.00 |
