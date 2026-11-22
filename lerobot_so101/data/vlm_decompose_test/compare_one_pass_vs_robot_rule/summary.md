# Decomposition: one-pass (original) vs two-pass + robot-parts rule

- one-pass (original): `vlm_decompose_test/2026-10-02_07-42-36/results.json`
- two-pass + robot-parts rule: `vlm_decompose_test/2026-10-02_09-31-34/results.json`
- 195 scene x prompt pairs present in both runs, one trial each

## Checks

| check | one-pass (original) | two-pass + robot-parts rule | change | p (exact McNemar) |
|---|---|---|---|---|
| all three objects identified | 0.53 | 0.73 | +0.20 | 2.7e-07 |
| right objects planned | 0.58 | 0.71 | +0.13 | 0.0013 |
| order respected | 0.96 | 0.95 | -0.01 | 1 |
| right number of sub-tasks | 0.59 | 0.65 | +0.06 | 0.15 |
| destination is the tray | 0.97 | 0.98 | +0.02 | 0.51 |
| overall correct | 0.49 | 0.59 | +0.10 | 0.023 |

## Overall correct by category

| category | n | one-pass (original) | two-pass + robot-parts rule | change |
|---|---|---|---|---|
| base | 39 | 0.56 | 0.56 | +0.00 |
| omission | 39 | 0.46 | 0.49 | +0.03 |
| sequencing | 39 | 0.62 | 0.69 | +0.08 |
| conditional | 39 | 0.21 | 0.51 | +0.31 |
| quantity | 39 | 0.62 | 0.69 | +0.08 |

## Case by case (same pairs)

Exact McNemar test on the 63 pairs where the two disagree: p = 0.023.

| outcome | pairs |
|---|---|
| both correct | 74 |
| only two-pass + robot-parts rule correct | 41 |
| only one-pass (original) correct | 22 |
| both wrong | 58 |

## Failure modes (wrong cases)

| failure mode | one-pass (original) | two-pass + robot-parts rule |
|---|---|---|
| missed_object | 92 | 47 |
| planned_fixture | 0 | 15 |
| wrong_count | 0 | 2 |
| wrong_objects | 7 | 16 |

## Objects missed by the perception step (cases)

| object | one-pass (original) | two-pass + robot-parts rule |
|---|---|---|
| banana | 0 | 0 |
| toy | 8 | 0 |
| pouch | 89 | 53 |

## Overall correct by prompt

| category | prompt | n | one-pass (original) | two-pass + robot-parts rule | change |
|---|---|---|---|---|---|
| base | put everything on the tray | 39 | 0.56 | 0.56 | +0.00 |
| omission | put everything on the tray except the banana | 7 | 0.14 | 0.14 | +0.00 |
| omission | put everything on the tray except the toy | 7 | 0.00 | 0.57 | +0.57 |
| omission | put everything on the tray except the pouch | 7 | 0.86 | 0.43 | -0.43 |
| omission | put everything on the tray except the banana and the toy | 6 | 0.00 | 0.83 | +0.83 |
| omission | put everything on the tray but leave the toy and the pouch | 6 | 1.00 | 0.67 | -0.33 |
| omission | put everything on the tray, don't touch the pouch | 6 | 0.83 | 0.33 | -0.50 |
| sequencing | put the banana on the tray first, then the toy, then the pouch | 8 | 1.00 | 0.62 | -0.38 |
| sequencing | put the pouch on the tray first, then the banana, then the toy | 8 | 1.00 | 0.88 | -0.12 |
| sequencing | put everything on the tray, starting with the toy | 8 | 0.12 | 0.62 | +0.50 |
| sequencing | put everything on the tray, but the pouch goes last | 8 | 0.88 | 0.62 | -0.25 |
| sequencing | put everything on the tray, put the banana after the toy | 7 | 0.00 | 0.71 | +0.71 |
| conditional | put the fruit on the tray | 7 | 0.29 | 0.43 | +0.14 |
| conditional | put the brown objects on the tray | 7 | 0.00 | 0.29 | +0.29 |
| conditional | put everything that is not edible on the tray | 7 | 0.29 | 0.57 | +0.29 |
| conditional | put only the soft, furry things on the tray | 6 | 0.67 | 1.00 | +0.33 |
| conditional | if there is a banana on the table, put only the banana on the tray; otherwise put everything on the tray | 6 | 0.00 | 0.00 | +0.00 |
| conditional | if there is an apple on the table, put the apple on the tray; otherwise put the toy on the tray | 6 | 0.00 | 0.83 | +0.83 |
| quantity | put one object on the tray | 8 | 0.62 | 0.88 | +0.25 |
| quantity | put two objects on the tray | 8 | 0.62 | 0.88 | +0.25 |
| quantity | put all three objects on the tray | 8 | 0.62 | 0.75 | +0.12 |
| quantity | put one of the brown objects on the tray | 8 | 0.62 | 0.12 | -0.50 |
| quantity | put two objects on the tray, but not the banana | 7 | 0.57 | 0.86 | +0.29 |
