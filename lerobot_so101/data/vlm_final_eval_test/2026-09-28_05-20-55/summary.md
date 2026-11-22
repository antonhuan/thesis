# VLM final-evaluation test

- scenes: 49 from `vlm_final_eval_scenes.json`
- model: `Qwen/Qwen3-VL-8B-Instruct` (temp 0.1), two-pass survey -> judge
- trials per prompt per scene: 1
- unscored cases (hinge on a partial object / name an absent one): 0

false neg = done but judged incomplete (needless retry round); false pos = not done but judged complete (task stops early).

## Per category

| category | n | correct | false neg | false pos | survey ok | judge consistent |
|---|---|---|---|---|---|---|
| base | 49 | 0.98 | 1/9 | 0/40 | 0.61 | 1.00 |
| omission | 286 | 0.77 | 14/44 | 52/242 | 0.59 | 0.82 |
| sequencing | 236 | 0.85 | 22/36 | 13/200 | 0.59 | 0.92 |
| conditional | 280 | 0.69 | 24/44 | 62/236 | 0.59 | 0.79 |
| quantity | 231 | 0.68 | 31/63 | 43/168 | 0.58 | 0.77 |
| **all** | 1082 | 0.76 | 92/196 | 170/886 | 0.59 | 0.83 |

## Per prompt

| category | prompt | n | correct | false neg | false pos | survey ok | judge consistent |
|---|---|---|---|---|---|---|---|
| base | put everything on the tray | 49 | 0.98 | 1/9 | 0/40 | 0.61 | 1.00 |
| omission | put everything on the tray except the banana | 44 | 0.89 | 3/6 | 2/38 | 0.59 | 0.98 |
| omission | put everything on the tray except the toy | 49 | 0.84 | 0/0 | 8/49 | 0.61 | 0.84 |
| omission | put everything on the tray except the pouch | 49 | 0.67 | 3/18 | 13/31 | 0.59 | 0.63 |
| omission | put everything on the tray except the banana and the toy | 46 | 0.89 | 0/0 | 5/46 | 0.59 | 0.88 |
| omission | put everything on the tray but leave the toy and the pouch | 49 | 0.65 | 2/2 | 15/47 | 0.59 | 0.66 |
| omission | put everything on the tray, don't touch the pouch | 49 | 0.69 | 6/18 | 9/31 | 0.59 | 0.89 |
| sequencing | put the banana on the tray first, then the toy, then the pouch | 46 | 0.89 | 5/6 | 0/40 | 0.59 | 1.00 |
| sequencing | put the pouch on the tray first, then the banana, then the toy | 46 | 0.87 | 6/6 | 0/40 | 0.61 | 0.97 |
| sequencing | put everything on the tray, starting with the toy | 49 | 0.98 | 1/9 | 0/40 | 0.59 | 1.00 |
| sequencing | put everything on the tray, but the pouch goes last | 49 | 0.67 | 7/9 | 9/40 | 0.59 | 0.68 |
| sequencing | put everything on the tray, put the banana after the toy | 46 | 0.85 | 3/6 | 4/40 | 0.59 | 0.91 |
| conditional | put the fruit on the tray | 46 | 0.80 | 2/2 | 7/44 | 0.59 | 0.84 |
| conditional | put the brown objects on the tray | 47 | 0.66 | 7/9 | 9/38 | 0.57 | 0.74 |
| conditional | put everything that is not edible on the tray | 47 | 0.68 | 4/9 | 11/38 | 0.57 | 0.83 |
| conditional | put only the soft, furry things on the tray | 47 | 0.77 | 2/11 | 9/36 | 0.57 | 0.87 |
| conditional | if there is a banana on the table, put only the banana on the tray; otherwise put everything on the tray | 46 | 0.83 | 2/2 | 6/44 | 0.61 | 0.91 |
| conditional | if there is an apple on the table, put the apple on the tray; otherwise put the toy on the tray | 47 | 0.43 | 7/11 | 20/36 | 0.60 | 0.55 |
| quantity | put one object on the tray | 47 | 0.36 | 0/13 | 30/34 | 0.60 | 0.57 |
| quantity | put two objects on the tray | 47 | 0.62 | 13/27 | 5/20 | 0.57 | 0.93 |
| quantity | put all three objects on the tray | 46 | 0.96 | 2/6 | 0/40 | 0.59 | 1.00 |
| quantity | put one of the brown objects on the tray | 47 | 0.68 | 11/11 | 4/36 | 0.60 | 0.51 |
| quantity | put two objects on the tray, but not the banana | 44 | 0.80 | 5/6 | 4/38 | 0.57 | 0.94 |

## Per scene

| scene | banana | toy | pouch | base | omission | sequencing | conditional | quantity | all | survey ok | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-19_03-51-37_01 | table | tray | table | 1.00 | 0.83 | 1.00 | 0.67 | 0.80 | 0.83 | 1.00 |  |
| 2026-08-19_04-10-27_01 | tray | tray | table | 1.00 | 1.00 | 0.80 | 0.83 | 0.80 | 0.87 | 1.00 | banana lies along the tray's back rim |
| 2026-08-19_04-21-15_04 | tray | tray | table | 1.00 | 0.67 | 1.00 | 0.50 | 0.40 | 0.65 | 0.00 | pouch end touches the tray rim, body on the table |
| 2026-08-19_04-21-15_07 | tray | table | table | 1.00 | 0.50 | 1.00 | 0.17 | 1.00 | 0.65 | 0.00 | toy on the table behind the pouch |
| 2026-08-19_04-21-15_07_r2 | tray | table | table | 1.00 | 0.33 | 1.00 | 0.33 | 0.80 | 0.61 | 0.00 | toy on the table behind the pouch |
| 2026-08-19_04-44-51_01 | table | tray | tray | 1.00 | 0.50 | 1.00 | 0.33 | 0.40 | 0.57 | 0.00 | toy lying on the pouch inside the tray |
| 2026-08-19_04-44-51_01_r2 | table | tray | tray | 1.00 | 0.83 | 1.00 | 0.50 | 0.40 | 0.70 | 0.00 | toy lying on the pouch inside the tray |
| 2026-08-19_05-50-13_01 | table | tray | table | 1.00 | 0.67 | 1.00 | 0.67 | 0.80 | 0.78 | 1.00 |  |
| 2026-08-19_05-50-13_01_r2 | table | tray | table | 1.00 | 0.67 | 1.00 | 0.67 | 0.80 | 0.78 | 1.00 | toy overhangs the tray's back rim |
| 2026-08-19_05-50-13_02 | table | tray | table | 1.00 | 0.67 | 1.00 | 0.67 | 0.80 | 0.78 | 1.00 |  |
| 2026-08-19_05-50-13_04 | table | tray | table | 1.00 | 0.83 | 1.00 | 0.83 | 0.80 | 0.87 | 1.00 |  |
| 2026-08-19_05-50-13_04_r2 | table | tray | table | 1.00 | 0.83 | 1.00 | 0.67 | 0.80 | 0.83 | 1.00 |  |
| 2026-08-19_06-31-10_01 | table | tray | table | 1.00 | 1.00 | 1.00 | 0.67 | 0.80 | 0.87 | 0.00 |  |
| 2026-08-19_06-57-56_02 | table | tray | table | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.96 | 1.00 |  |
| 2026-08-19_06-57-56_02_r2 | table | tray | tray | 1.00 | 0.67 | 1.00 | 0.67 | 0.60 | 0.74 | 1.00 | pouch across the front of the tray, toy behind it |
| 2026-08-19_07-17-03_01 | table | tray | table | 1.00 | 1.00 | 1.00 | 0.83 | 0.80 | 0.91 | 1.00 |  |
| 2026-08-19_07-17-03_01_r2 | table | tray | partial | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 0.93 | 0.86 | pouch straddles the tray's left rim |
| 2026-08-20_06-23-01_01 | tray | tray | table | 1.00 | 1.00 | 0.80 | 0.67 | 0.60 | 0.78 | 1.00 |  |
| 2026-08-20_06-23-01_02 | table | tray | table | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.96 | 1.00 |  |
| 2026-08-20_06-23-01_02_r2 | table | tray | tray | 1.00 | 0.83 | 1.00 | 0.67 | 0.40 | 0.74 | 0.00 | toy on top of the pouch, overhanging the back rim |
| 2026-08-20_06-23-01_03 | tray | tray | table | 1.00 | 0.67 | 1.00 | 0.83 | 0.60 | 0.78 | 0.00 |  |
| 2026-08-20_06-49-37_01 | tray | tray | table | 1.00 | 0.83 | 1.00 | 0.67 | 0.80 | 0.83 | 1.00 |  |
| 2026-08-20_07-24-04_01 | tray | tray | table | 1.00 | 0.67 | 1.00 | 0.83 | 0.80 | 0.83 | 1.00 |  |
| 2026-08-20_07-24-04_01_r2 | tray | tray | tray | 0.00 | 0.50 | 0.00 | 0.67 | 0.00 | 0.30 | 0.00 |  |
| 2026-08-20_07-24-04_03 | tray | tray | tray | 1.00 | 0.67 | 0.60 | 0.50 | 0.60 | 0.61 | 0.00 | banana partly under the pouch |
| 2026-08-20_08-03-31_02 | tray | tray | table | 1.00 | 1.00 | 0.60 | 0.83 | 0.80 | 0.83 | 1.00 |  |
| 2026-08-20_08-03-31_03 | tray | tray | table | 1.00 | 1.00 | 1.00 | 0.67 | 0.80 | 0.87 | 1.00 | pouch partly hidden behind the arm |
| 2026-09-02_09-11-20_02 | tray | tray | table | 1.00 | 0.83 | 1.00 | 0.83 | 0.60 | 0.83 | 0.43 |  |
| 2026-09-02_09-41-46_03 | table | table | table | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | operator's hand in frame; banana resting on the pouch; tray empty |
| 2026-09-02_09-41-46_04 | tray | tray | table | 1.00 | 1.00 | 0.80 | 1.00 | 0.80 | 0.91 | 1.00 |  |
| 2026-09-02_10-25-12_01 | tray | tray | table | 1.00 | 1.00 | 0.80 | 1.00 | 0.80 | 0.91 | 1.00 |  |
| 2026-09-02_10-37-10_01 | tray | tray | table | 1.00 | 1.00 | 0.60 | 0.83 | 0.80 | 0.83 | 1.00 |  |
| 2026-09-03_06-00-35_01 | tray | tray | table | 1.00 | 1.00 | 0.60 | 0.83 | 0.80 | 0.83 | 1.00 |  |
| 2026-09-03_06-00-35_01_r2 | tray | tray | table | 1.00 | 1.00 | 0.60 | 0.83 | 0.80 | 0.83 | 1.00 |  |
| 2026-09-03_07-09-01_01 | table | tray | tray | 1.00 | 0.50 | 1.00 | 0.33 | 0.40 | 0.57 | 0.00 |  |
| 2026-09-03_07-09-01_02 | tray | tray | tray | 1.00 | 1.00 | 0.80 | 0.83 | 0.60 | 0.83 | 1.00 | toy partly out of frame |
| 2026-09-06_05-50-19_01 | absent | tray | tray | 1.00 | 0.75 | 0.50 | 1.00 | 0.67 | 0.79 | 1.00 | two-object run (no banana) |
| 2026-09-06_05-53-28_01 | absent | tray | tray | 1.00 | 0.50 | 0.50 | 0.50 | 0.33 | 0.50 | 1.00 | two-object run (no banana) |
| 2026-09-09_02-42-03_01 | absent | tray | tray | 1.00 | 0.25 | 0.50 | 0.25 | 0.33 | 0.36 | 0.00 | two-object run (no banana) |
| 2026-09-09_07-16-05_01 | tray | tray | table | 1.00 | 0.50 | 1.00 | 1.00 | 0.80 | 0.83 | 0.00 |  |
| 2026-09-10_02-49-37_01 | tray | tray | table | 1.00 | 0.83 | 0.80 | 0.83 | 0.80 | 0.83 | 1.00 |  |
| 2026-09-10_03-02-18_01 | table | tray | partial | 1.00 | 0.60 | 1.00 | 1.00 | 1.00 | 0.86 | 1.00 | pouch's right end over the tray rim, most of it on the table |
| 2026-09-10_04-41-17_01 | tray | tray | table | 1.00 | 0.67 | 1.00 | 1.00 | 0.80 | 0.87 | 0.00 |  |
| 2026-09-16_08-18-24_01 | tray | tray | table | 1.00 | 0.67 | 1.00 | 0.67 | 0.60 | 0.74 | 0.00 |  |
| 2026-09-16_08-18-24_01_r2 | tray | tray | tray | 1.00 | 0.83 | 0.20 | 0.50 | 0.60 | 0.57 | 0.00 | banana under the pouch, only its ends visible |
| 2026-09-16_08-31-16_01 | tray | tray | tray | 1.00 | 0.83 | 0.40 | 0.17 | 0.60 | 0.52 | 0.00 | banana on top of the toy |
| 2026-09-16_08-31-16_02 | tray | tray | tray | 1.00 | 0.50 | 0.20 | 0.33 | 0.20 | 0.35 | 0.00 | banana placed first, fully hidden under the pouch/toy |
| 2026-09-17_06-54-19_01 | table | tray | table | 1.00 | 0.67 | 1.00 | 0.50 | 0.80 | 0.74 | 1.00 |  |
| 2026-09-17_06-54-19_01_r2 | table | tray | tray | 1.00 | 0.50 | 1.00 | 0.67 | 0.60 | 0.70 | 0.00 | toy on top of the pouch |

## Error attribution

| cause | cases | share | meaning |
|---|---|---|---|
| perception | 158 | 60% | the survey mislocated, missed or merged an object |
| judge | 104 | 40% | the survey was right; the judge misread the instruction against it |

Perception errors by object: pouch 101, banana 84, toy 37

## Survey side effects

- surveys listing a fixture (tray / arm / table / hand) as an object: 340/1082
- retry targets right (remaining_from_survey == objects still on the table), over cases judged incomplete: 0.69 (n=783)
