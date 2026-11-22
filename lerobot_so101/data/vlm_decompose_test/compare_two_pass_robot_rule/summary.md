# Decomposition: two-pass vs two-pass + robot-parts rule

- two-pass: `vlm_decompose_test/2026-10-02_07-42-16/results.json`
- two-pass + robot-parts rule: `vlm_decompose_test/2026-10-02_09-31-34/results.json`
- 195 scene x prompt pairs present in both runs, one trial each

## Checks

| check | two-pass | two-pass + robot-parts rule | change | p (exact McNemar) |
|---|---|---|---|---|
| all three objects identified | 0.73 | 0.73 | +0.00 | 1 |
| right objects planned | 0.71 | 0.71 | +0.00 | 1 |
| order respected | 0.95 | 0.95 | +0.00 | 1 |
| right number of sub-tasks | 0.62 | 0.65 | +0.03 | 0.11 |
| destination is the tray | 0.98 | 0.98 | +0.00 | 1 |
| overall correct | 0.56 | 0.59 | +0.03 | 0.18 |

## Overall correct by category

| category | n | two-pass | two-pass + robot-parts rule | change |
|---|---|---|---|---|
| base | 39 | 0.46 | 0.56 | +0.10 |
| omission | 39 | 0.44 | 0.49 | +0.05 |
| sequencing | 39 | 0.67 | 0.69 | +0.03 |
| conditional | 39 | 0.56 | 0.51 | -0.05 |
| quantity | 39 | 0.69 | 0.69 | +0.00 |

## Case by case (same pairs)

Exact McNemar test on the 9 pairs where the two disagree: p = 0.180.

| outcome | pairs |
|---|---|
| both correct | 108 |
| only two-pass + robot-parts rule correct | 7 |
| only two-pass correct | 2 |
| both wrong | 78 |

## Failure modes (wrong cases)

| failure mode | two-pass | two-pass + robot-parts rule |
|---|---|---|
| missed_object | 42 | 47 |
| planned_fixture | 27 | 15 |
| wrong_count | 2 | 2 |
| wrong_objects | 14 | 16 |

## Objects missed by the perception step (cases)

| object | two-pass | two-pass + robot-parts rule |
|---|---|---|
| banana | 0 | 0 |
| toy | 0 | 0 |
| pouch | 53 | 53 |

## Overall correct by prompt

| category | prompt | n | two-pass | two-pass + robot-parts rule | change |
|---|---|---|---|---|---|
| base | put everything on the tray | 39 | 0.46 | 0.56 | +0.10 |
| omission | put everything on the tray except the banana | 7 | 0.14 | 0.14 | +0.00 |
| omission | put everything on the tray except the toy | 7 | 0.57 | 0.57 | +0.00 |
| omission | put everything on the tray except the pouch | 7 | 0.43 | 0.43 | +0.00 |
| omission | put everything on the tray except the banana and the toy | 6 | 0.83 | 0.83 | +0.00 |
| omission | put everything on the tray but leave the toy and the pouch | 6 | 0.50 | 0.67 | +0.17 |
| omission | put everything on the tray, don't touch the pouch | 6 | 0.17 | 0.33 | +0.17 |
| sequencing | put the banana on the tray first, then the toy, then the pouch | 8 | 0.62 | 0.62 | +0.00 |
| sequencing | put the pouch on the tray first, then the banana, then the toy | 8 | 0.88 | 0.88 | +0.00 |
| sequencing | put everything on the tray, starting with the toy | 8 | 0.62 | 0.62 | +0.00 |
| sequencing | put everything on the tray, but the pouch goes last | 8 | 0.50 | 0.62 | +0.12 |
| sequencing | put everything on the tray, put the banana after the toy | 7 | 0.71 | 0.71 | +0.00 |
| conditional | put the fruit on the tray | 7 | 0.43 | 0.43 | +0.00 |
| conditional | put the brown objects on the tray | 7 | 0.57 | 0.29 | -0.29 |
| conditional | put everything that is not edible on the tray | 7 | 0.57 | 0.57 | +0.00 |
| conditional | put only the soft, furry things on the tray | 6 | 1.00 | 1.00 | +0.00 |
| conditional | if there is a banana on the table, put only the banana on the tray; otherwise put everything on the tray | 6 | 0.00 | 0.00 | +0.00 |
| conditional | if there is an apple on the table, put the apple on the tray; otherwise put the toy on the tray | 6 | 0.83 | 0.83 | +0.00 |
| quantity | put one object on the tray | 8 | 0.88 | 0.88 | +0.00 |
| quantity | put two objects on the tray | 8 | 0.88 | 0.88 | +0.00 |
| quantity | put all three objects on the tray | 8 | 0.75 | 0.75 | +0.00 |
| quantity | put one of the brown objects on the tray | 8 | 0.12 | 0.12 | +0.00 |
| quantity | put two objects on the tray, but not the banana | 7 | 0.86 | 0.86 | +0.00 |
