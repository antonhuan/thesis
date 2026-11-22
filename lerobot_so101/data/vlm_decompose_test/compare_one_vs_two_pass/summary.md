# Decomposition: one-pass (original) vs two-pass

- one-pass (original): `vlm_decompose_test/2026-10-01_02-52-25/results.json`
- two-pass: `vlm_decompose_test/2026-09-27_03-43-57_sample200/results.json`
- 200 scene x prompt pairs present in both runs, one trial each

## Checks

| check | one-pass (original) | two-pass | change | p (exact McNemar) |
|---|---|---|---|---|
| all three objects identified | 0.50 | 0.71 | +0.21 | 6.1e-07 |
| right objects planned | 0.57 | 0.68 | +0.11 | 0.013 |
| order respected | 0.95 | 0.94 | -0.01 | 0.77 |
| right number of sub-tasks | 0.58 | 0.62 | +0.04 | 0.49 |
| destination is the tray | 0.95 | 0.97 | +0.02 | 0.55 |
| overall correct | 0.45 | 0.55 | +0.10 | 0.038 |

## Overall correct by category

| category | n | one-pass (original) | two-pass | change |
|---|---|---|---|---|
| base | 9 | 0.67 | 0.67 | +0.00 |
| omission | 54 | 0.54 | 0.43 | -0.11 |
| sequencing | 45 | 0.58 | 0.69 | +0.11 |
| conditional | 52 | 0.19 | 0.46 | +0.27 |
| quantity | 40 | 0.47 | 0.65 | +0.18 |

## Case by case (same pairs)

Exact McNemar test on the 84 pairs where the two disagree: p = 0.038.

| outcome | pairs |
|---|---|
| both correct | 58 |
| only two-pass correct | 52 |
| only one-pass (original) correct | 32 |
| both wrong | 58 |

## Failure modes (wrong cases)

| failure mode | one-pass (original) | two-pass |
|---|---|---|
| missed_object | 100 | 44 |
| planned_fixture | 0 | 25 |
| wrong_count | 0 | 1 |
| wrong_objects | 10 | 20 |

## Objects missed by the perception step (cases)

| object | one-pass (original) | two-pass |
|---|---|---|
| banana | 0 | 0 |
| toy | 12 | 0 |
| pouch | 97 | 57 |

## Overall correct by prompt

| category | prompt | n | one-pass (original) | two-pass | change |
|---|---|---|---|---|---|
| base | put everything on the tray | 9 | 0.67 | 0.67 | +0.00 |
| omission | put everything on the tray except the banana | 9 | 0.33 | 0.22 | -0.11 |
| omission | put everything on the tray except the toy | 9 | 0.00 | 0.67 | +0.67 |
| omission | put everything on the tray except the pouch | 9 | 1.00 | 0.44 | -0.56 |
| omission | put everything on the tray except the banana and the toy | 9 | 0.00 | 0.78 | +0.78 |
| omission | put everything on the tray but leave the toy and the pouch | 9 | 1.00 | 0.33 | -0.67 |
| omission | put everything on the tray, don't touch the pouch | 9 | 0.89 | 0.11 | -0.78 |
| sequencing | put the banana on the tray first, then the toy, then the pouch | 9 | 1.00 | 0.67 | -0.33 |
| sequencing | put the pouch on the tray first, then the banana, then the toy | 9 | 1.00 | 0.67 | -0.33 |
| sequencing | put everything on the tray, starting with the toy | 9 | 0.11 | 0.67 | +0.56 |
| sequencing | put everything on the tray, but the pouch goes last | 9 | 0.78 | 0.67 | -0.11 |
| sequencing | put everything on the tray, put the banana after the toy | 9 | 0.00 | 0.78 | +0.78 |
| conditional | put the fruit on the tray | 9 | 0.11 | 0.56 | +0.44 |
| conditional | put the brown objects on the tray | 9 | 0.00 | 0.44 | +0.44 |
| conditional | put everything that is not edible on the tray | 9 | 0.33 | 0.22 | -0.11 |
| conditional | put only the soft, furry things on the tray | 9 | 0.67 | 0.67 | +0.00 |
| conditional | if there is a banana on the table, put only the banana on the tray; otherwise put everything on the tray | 8 | 0.00 | 0.00 | +0.00 |
| conditional | if there is an apple on the table, put the apple on the tray; otherwise put the toy on the tray | 8 | 0.00 | 0.88 | +0.88 |
| quantity | put one object on the tray | 8 | 0.62 | 0.75 | +0.12 |
| quantity | put two objects on the tray | 8 | 0.62 | 0.88 | +0.25 |
| quantity | put all three objects on the tray | 8 | 0.38 | 0.62 | +0.25 |
| quantity | put one of the brown objects on the tray | 8 | 0.50 | 0.38 | -0.12 |
| quantity | put two objects on the tray, but not the banana | 8 | 0.25 | 0.62 | +0.38 |
