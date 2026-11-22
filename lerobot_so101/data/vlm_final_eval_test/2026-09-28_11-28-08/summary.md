# VLM final-evaluation test

- scenes: 49 from `vlm_final_eval_scenes.json`
- model: `Qwen/Qwen3-VL-8B-Instruct` (temp 0.1), two-pass survey -> judge — survey sees BEFORE (round-1 decompose frame) + AFTER — object inventory from BEFORE (monitor grounder), judge sees appearance
- trials per prompt per scene: 1 — fixed subset of 200 cases (seed 0)
- unscored cases (hinge on a partial object / name an absent one): 0

false neg = done but judged incomplete (needless retry round); false pos = not done but judged complete (task stops early).

## Per category

| category | n | correct | false neg | false pos | survey ok | judge consistent |
|---|---|---|---|---|---|---|
| base | 9 | 0.89 | 0/4 | 1/5 | 0.89 | 1.00 |
| omission | 54 | 0.61 | 5/14 | 16/40 | 0.37 | 0.72 |
| sequencing | 45 | 0.62 | 10/20 | 7/25 | 0.56 | 0.73 |
| conditional | 52 | 0.75 | 2/20 | 11/32 | 0.40 | 0.84 |
| quantity | 40 | 0.65 | 4/20 | 10/20 | 0.42 | 0.71 |
| **all** | 200 | 0.67 | 21/78 | 45/122 | 0.46 | 0.77 |

## Per prompt

| category | prompt | n | correct | false neg | false pos | survey ok | judge consistent |
|---|---|---|---|---|---|---|---|
| base | put everything on the tray | 9 | 0.89 | 0/4 | 1/5 | 0.89 | 1.00 |
| omission | put everything on the tray except the banana | 9 | 0.44 | 2/4 | 3/5 | 0.56 | 0.75 |
| omission | put everything on the tray except the toy | 9 | 0.78 | 0/0 | 2/9 | 0.11 | 0.89 |
| omission | put everything on the tray except the pouch | 9 | 0.33 | 1/4 | 5/5 | 0.11 | 1.00 |
| omission | put everything on the tray except the banana and the toy | 9 | 0.67 | 0/0 | 3/9 | 0.44 | 0.62 |
| omission | put everything on the tray but leave the toy and the pouch | 9 | 0.89 | 0/2 | 1/7 | 0.56 | 0.75 |
| omission | put everything on the tray, don't touch the pouch | 9 | 0.56 | 2/4 | 2/5 | 0.44 | 0.50 |
| sequencing | put the banana on the tray first, then the toy, then the pouch | 9 | 0.56 | 2/4 | 2/5 | 0.56 | 0.60 |
| sequencing | put the pouch on the tray first, then the banana, then the toy | 9 | 0.67 | 3/4 | 0/5 | 0.56 | 0.80 |
| sequencing | put everything on the tray, starting with the toy | 9 | 0.89 | 0/4 | 1/5 | 0.67 | 0.89 |
| sequencing | put everything on the tray, but the pouch goes last | 9 | 0.33 | 4/4 | 2/5 | 0.56 | 0.20 |
| sequencing | put everything on the tray, put the banana after the toy | 9 | 0.67 | 1/4 | 2/5 | 0.44 | 0.89 |
| conditional | put the fruit on the tray | 9 | 0.78 | 0/2 | 2/7 | 0.00 | 0.78 |
| conditional | put the brown objects on the tray | 9 | 0.44 | 1/4 | 4/5 | 0.22 | 0.86 |
| conditional | put everything that is not edible on the tray | 9 | 0.89 | 0/4 | 1/5 | 0.78 | 1.00 |
| conditional | put only the soft, furry things on the tray | 9 | 1.00 | 0/4 | 0/5 | 0.44 | 1.00 |
| conditional | if there is a banana on the table, put only the banana on the tray; otherwise put everything on the tray | 8 | 1.00 | 0/2 | 0/6 | 0.50 | 1.00 |
| conditional | if there is an apple on the table, put the apple on the tray; otherwise put the toy on the tray | 8 | 0.38 | 1/4 | 4/4 | 0.50 | 0.38 |
| quantity | put one object on the tray | 8 | 0.50 | 0/4 | 4/4 | 0.38 | 0.50 |
| quantity | put two objects on the tray | 8 | 0.75 | 1/4 | 1/4 | 0.25 | 0.86 |
| quantity | put all three objects on the tray | 8 | 0.75 | 1/4 | 1/4 | 0.50 | 1.00 |
| quantity | put one of the brown objects on the tray | 8 | 0.50 | 0/4 | 4/4 | 0.38 | 0.50 |
| quantity | put two objects on the tray, but not the banana | 8 | 0.75 | 2/4 | 0/4 | 0.62 | 0.83 |

## Per scene

| scene | banana | toy | pouch | base | omission | sequencing | conditional | quantity | all | survey ok | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-19_03-51-37_01 | table | tray | table | 1.00 | - | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| 2026-08-19_04-10-27_01 | tray | tray | table | 1.00 | - | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | banana lies along the tray's back rim |
| 2026-08-19_04-21-15_04 | tray | tray | table | - | 1.00 | - | 1.00 | 0.00 | 0.75 | 1.00 | pouch end touches the tray rim, body on the table |
| 2026-08-19_04-21-15_07 | tray | table | table | - | 0.50 | - | 1.00 | - | 0.75 | 0.00 | toy on the table behind the pouch |
| 2026-08-19_04-21-15_07_r2 | tray | table | table | - | 0.50 | - | 1.00 | - | 0.75 | 0.00 | toy on the table behind the pouch |
| 2026-08-19_04-44-51_01 | table | tray | tray | - | 0.00 | 1.00 | 1.00 | 0.00 | 0.50 | 1.00 | toy lying on the pouch inside the tray |
| 2026-08-19_04-44-51_01_r2 | table | tray | tray | - | 1.00 | - | 1.00 | 1.00 | 1.00 | 1.00 | toy lying on the pouch inside the tray |
| 2026-08-19_05-50-13_01 | table | tray | table | - | 0.00 | 1.00 | 1.00 | - | 0.75 | 0.00 |  |
| 2026-08-19_05-50-13_01_r2 | table | tray | table | - | 0.00 | 1.00 | 0.50 | - | 0.50 | 0.00 | toy overhangs the tray's back rim |
| 2026-08-19_05-50-13_02 | table | tray | table | - | 0.00 | - | 1.00 | 1.00 | 0.75 | 0.00 |  |
| 2026-08-19_05-50-13_04 | table | tray | table | - | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |  |
| 2026-08-19_05-50-13_04_r2 | table | tray | table | - | 0.00 | - | 1.00 | 1.00 | 0.80 | 0.00 |  |
| 2026-08-19_06-31-10_01 | table | tray | table | - | 0.00 | 1.00 | 0.00 | 1.00 | 0.50 | 0.00 |  |
| 2026-08-19_06-57-56_02 | table | tray | table | - | 0.50 | - | 0.50 | - | 0.50 | 1.00 |  |
| 2026-08-19_06-57-56_02_r2 | table | tray | tray | - | 1.00 | 1.00 | 1.00 | 0.00 | 0.75 | 1.00 | pouch across the front of the tray, toy behind it |
| 2026-08-19_07-17-03_01 | table | tray | table | - | 1.00 | 1.00 | 0.00 | 1.00 | 0.80 | 0.00 |  |
| 2026-08-19_07-17-03_01_r2 | table | tray | partial | - | 1.00 | 1.00 | 1.00 | - | 1.00 | 1.00 | pouch straddles the tray's left rim |
| 2026-08-20_06-23-01_01 | tray | tray | table | - | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 | 1.00 |  |
| 2026-08-20_06-23-01_02 | table | tray | table | - | 0.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.00 |  |
| 2026-08-20_06-23-01_02_r2 | table | tray | tray | - | 1.00 | - | 0.00 | 0.00 | 0.40 | 0.00 | toy on top of the pouch, overhanging the back rim |
| 2026-08-20_06-23-01_03 | tray | tray | table | 1.00 | - | 1.00 | 1.00 | 0.00 | 0.75 | 1.00 |  |
| 2026-08-20_06-49-37_01 | tray | tray | table | - | 0.00 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 |  |
| 2026-08-20_07-24-04_01 | tray | tray | table | - | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.00 |  |
| 2026-08-20_07-24-04_01_r2 | tray | tray | tray | - | 1.00 | 0.67 | - | 1.00 | 0.80 | 0.00 |  |
| 2026-08-20_07-24-04_03 | tray | tray | tray | 1.00 | - | 0.67 | - | - | 0.75 | 1.00 | banana partly under the pouch |
| 2026-08-20_08-03-31_02 | tray | tray | table | - | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |  |
| 2026-08-20_08-03-31_03 | tray | tray | table | - | 0.00 | - | 0.50 | 0.00 | 0.25 | 0.00 | pouch partly hidden behind the arm |
| 2026-09-02_09-11-20_02 | tray | tray | table | - | 0.50 | - | 0.00 | 1.00 | 0.50 | 0.00 |  |
| 2026-09-02_09-41-46_03 | table | table | table | - | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | operator's hand in frame; banana resting on the pouch; tray empty |
| 2026-09-02_09-41-46_04 | tray | tray | table | - | 1.00 | 1.00 | 0.00 | 1.00 | 0.75 | 1.00 |  |
| 2026-09-02_10-25-12_01 | tray | tray | table | - | 1.00 | 0.00 | - | 0.50 | 0.50 | 0.00 |  |
| 2026-09-02_10-37-10_01 | tray | tray | table | - | 0.50 | - | 1.00 | 1.00 | 0.75 | 1.00 |  |
| 2026-09-03_06-00-35_01 | tray | tray | table | - | 1.00 | - | 0.50 | 0.00 | 0.50 | 0.00 |  |
| 2026-09-03_06-00-35_01_r2 | tray | tray | table | - | 1.00 | - | 1.00 | - | 1.00 | 0.00 |  |
| 2026-09-03_07-09-01_01 | table | tray | tray | - | 0.00 | 1.00 | 1.00 | 1.00 | 0.75 | 0.50 |  |
| 2026-09-03_07-09-01_02 | tray | tray | tray | - | 0.50 | 0.50 | - | 1.00 | 0.60 | 1.00 | toy partly out of frame |
| 2026-09-06_05-50-19_01 | absent | tray | tray | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.00 | 1.00 | two-object run (no banana) |
| 2026-09-06_05-53-28_01 | absent | tray | tray | - | 1.00 | 0.50 | - | - | 0.67 | 0.00 | two-object run (no banana) |
| 2026-09-09_02-42-03_01 | absent | tray | tray | 1.00 | - | 0.50 | 1.00 | - | 0.75 | 1.00 | two-object run (no banana) |
| 2026-09-09_07-16-05_01 | tray | tray | table | 1.00 | - | 0.00 | 1.00 | - | 0.67 | 1.00 |  |
| 2026-09-10_02-49-37_01 | tray | tray | table | - | 1.00 | 0.00 | 0.00 | - | 0.33 | 0.00 |  |
| 2026-09-10_03-02-18_01 | table | tray | partial | - | 0.00 | 0.00 | 1.00 | - | 0.33 | 1.00 | pouch's right end over the tray rim, most of it on the table |
| 2026-09-10_04-41-17_01 | tray | tray | table | 0.00 | 1.00 | - | 1.00 | 0.00 | 0.50 | 0.00 |  |
| 2026-09-16_08-18-24_01 | tray | tray | table | - | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.00 |  |
| 2026-09-16_08-18-24_01_r2 | tray | tray | tray | - | 0.50 | 0.50 | - | 0.00 | 0.40 | 0.00 | banana under the pouch, only its ends visible |
| 2026-09-16_08-31-16_01 | tray | tray | tray | 1.00 | - | 0.33 | - | 1.00 | 0.60 | 1.00 | banana on top of the toy |
| 2026-09-16_08-31-16_02 | tray | tray | tray | - | 1.00 | 0.00 | - | - | 0.50 | 0.00 | banana placed first, fully hidden under the pouch/toy |
| 2026-09-17_06-54-19_01 | table | tray | table | - | 0.00 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 |  |
| 2026-09-17_06-54-19_01_r2 | table | tray | tray | - | 0.00 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | toy on top of the pouch |

## Error attribution

| cause | cases | share | meaning |
|---|---|---|---|
| perception | 42 | 64% | the survey mislocated, missed or merged an object |
| judge | 24 | 36% | the survey was right; the judge misread the instruction against it |

Perception errors by object: pouch 42, toy 2, banana 2

## Survey side effects

- surveys listing a fixture (tray / arm / table / hand) as an object: 0/200
- retry targets right (remaining_from_survey == objects still on the table), over cases judged incomplete: 0.57 (n=94)
