# Decomposition: one pass vs two pass

- one pass: `vlm_decompose_test/2026-10-02_07-42-36_fixturefilter/results.json`
- two pass: `vlm_decompose_test/2026-10-02_11-35-25/results.json`
- 195 scene x prompt pairs present in both runs, one trial each

## Checks

| check | one pass | two pass | change | p (exact McNemar) |
|---|---|---|---|---|
| all three objects identified | 0.53 | 0.73 | +0.20 | 2.7e-07 |
| right objects planned | 0.58 | 0.72 | +0.14 | 0.00073 |
| order respected | 0.96 | 0.95 | -0.01 | 0.75 |
| right number of sub-tasks | 0.59 | 0.69 | +0.10 | 0.025 |
| destination is the tray | 0.97 | 1.00 | +0.03 | 0.031 |
| overall correct | 0.49 | 0.62 | +0.13 | 0.0041 |

## Overall correct by category

| category | n | one pass | two pass | change |
|---|---|---|---|---|
| base | 39 | 0.56 | 0.64 | +0.08 |
| omission | 39 | 0.46 | 0.51 | +0.05 |
| sequencing | 39 | 0.62 | 0.69 | +0.08 |
| conditional | 39 | 0.21 | 0.56 | +0.36 |
| quantity | 39 | 0.62 | 0.69 | +0.08 |

## Case by case (same pairs)

Exact McNemar test on the 71 pairs where the two disagree: p = 0.004.

| outcome | pairs |
|---|---|
| both correct | 73 |
| only two pass correct | 48 |
| only one pass correct | 23 |
| both wrong | 51 |

## Failure modes (wrong cases)

| failure mode | one pass | two pass |
|---|---|---|
| missed_object | 92 | 53 |
| wrong_count | 0 | 5 |
| wrong_objects | 7 | 16 |

## Objects missed by the perception step (cases)

| object | one pass | two pass |
|---|---|---|
| banana | 0 | 0 |
| toy | 8 | 0 |
| pouch | 89 | 53 |

## All three objects identified, by whether the instruction names the object

| object | one pass named | one pass not named | two pass named | two pass not named |
|---|---|---|---|---|
| banana | 0.47 (23/49) | 0.55 (80/146) | 0.76 (37/49) | 0.72 (105/146) |
| toy | 0.41 (23/56) | 0.58 (80/139) | 0.75 (42/56) | 0.72 (100/139) |
| pouch | 0.93 (40/43) | 0.41 (63/152) | 0.77 (33/43) | 0.72 (109/152) |

## Overall correct by prompt

| category | prompt | n | one pass | two pass | change |
|---|---|---|---|---|---|
| base | put everything on the tray | 39 | 0.56 | 0.64 | +0.08 |
| omission | put everything on the tray except the banana | 7 | 0.14 | 0.29 | +0.14 |
| omission | put everything on the tray except the toy | 7 | 0.00 | 0.57 | +0.57 |
| omission | put everything on the tray except the pouch | 7 | 0.86 | 0.29 | -0.57 |
| omission | put everything on the tray except the banana and the toy | 6 | 0.00 | 1.00 | +1.00 |
| omission | put everything on the tray but leave the toy and the pouch | 6 | 1.00 | 0.67 | -0.33 |
| omission | put everything on the tray, don't touch the pouch | 6 | 0.83 | 0.33 | -0.50 |
| sequencing | put the banana on the tray first, then the toy, then the pouch | 8 | 1.00 | 0.62 | -0.38 |
| sequencing | put the pouch on the tray first, then the banana, then the toy | 8 | 1.00 | 0.88 | -0.12 |
| sequencing | put everything on the tray, starting with the toy | 8 | 0.12 | 0.62 | +0.50 |
| sequencing | put everything on the tray, but the pouch goes last | 8 | 0.88 | 0.50 | -0.38 |
| sequencing | put everything on the tray, put the banana after the toy | 7 | 0.00 | 0.86 | +0.86 |
| conditional | put the fruit on the tray | 7 | 0.29 | 0.43 | +0.14 |
| conditional | put the brown objects on the tray | 7 | 0.00 | 0.57 | +0.57 |
| conditional | put everything that is not edible on the tray | 7 | 0.29 | 0.57 | +0.29 |
| conditional | put only the soft, furry things on the tray | 6 | 0.67 | 1.00 | +0.33 |
| conditional | if there is a banana on the table, put only the banana on the tray; otherwise put everything on the tray | 6 | 0.00 | 0.00 | +0.00 |
| conditional | if there is an apple on the table, put the apple on the tray; otherwise put the toy on the tray | 6 | 0.00 | 0.83 | +0.83 |
| quantity | put one object on the tray | 8 | 0.62 | 0.88 | +0.25 |
| quantity | put two objects on the tray | 8 | 0.62 | 0.88 | +0.25 |
| quantity | put all three objects on the tray | 8 | 0.62 | 0.75 | +0.12 |
| quantity | put one of the brown objects on the tray | 8 | 0.62 | 0.12 | -0.50 |
| quantity | put two objects on the tray, but not the banana | 7 | 0.57 | 0.86 | +0.29 |
