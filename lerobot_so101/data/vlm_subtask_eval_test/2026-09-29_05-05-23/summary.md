# VLM sub-task evaluation test

- attempts: `vlm_subtask_eval_attempts.json` — fixed subset of 200 (seed 0)
- model: `Qwen/Qwen3-VL-8B-Instruct` (temp 0.1); planner.evaluate on the saved clip, with the clip's fps and the monitor's stop reason, as the orchestrator calls it

false fail = object on the tray but judged failed (a needless retry); false pass = not on the tray but judged done (the task moves on without it).

## Overall

| verdict | n | correct | false fail | false pass |
|---|---|---|---|---|
| this run (pre-check, then evaluator) | 200 | 0.91 | 6/125 | 13/75 |
| evaluator alone (same run) | 200 | 0.88 | 24/125 | 1/75 |
| pre-check only when the arm did not move | 200 | 0.91 | 13/125 | 5/75 |
| pre-check alone, as 'object on the tray?' | 200 | 0.87 | 14/125 | 12/75 |
| logged at the time (older prompts/models) | 200 | 0.84 | 30/125 | 1/75 |

## By group

| group | n | correct | false fail | false pass |
|---|---|---|---|---|
| object moved this attempt | 175 | 0.92 | 1/100 | 13/75 |
| already on the tray, arm did not move | 25 | 0.80 | 5/25 | 0/0 |
| target: banana | 52 | 0.83 | 1/30 | 8/22 |
| target: toy | 81 | 0.98 | 1/68 | 1/13 |
| target: pouch | 67 | 0.88 | 4/27 | 4/40 |

## By sub-task text

| instruction | n | correct | false fail | false pass |
|---|---|---|---|---|
| put the banana on the tray | 46 | 0.83 | 1/25 | 7/21 |
| put the stuffed animal on the tray | 27 | 0.96 | 0/21 | 1/6 |
| put the toy on the tray | 27 | 1.00 | 0/24 | 0/3 |
| put the pouch on the tray | 25 | 0.80 | 3/17 | 2/8 |
| put the plush toy on the tray | 17 | 1.00 | 0/14 | 0/3 |
| put the brown pouch on the tray | 8 | 0.75 | 0/2 | 2/6 |
| put the roll on the tray | 5 | 1.00 | 0/1 | 0/4 |
| put the cylinder on the tray | 4 | 1.00 | 0/1 | 0/3 |
| put the bag on the tray | 4 | 1.00 | 0/2 | 0/2 |
| put the purse on the tray | 4 | 1.00 | 0/0 | 0/4 |
| put the cylindrical object on the tray | 4 | 1.00 | 0/0 | 0/4 |
| put the banana in the tray | 3 | 0.67 | 0/2 | 1/1 |
| put the banana in the box | 3 | 1.00 | 0/3 | 0/0 |
| put the plush toy in the tray | 3 | 1.00 | 0/2 | 0/1 |
| put the pouch in the tray | 3 | 1.00 | 0/2 | 0/1 |
| put the pencil case on the tray | 3 | 1.00 | 0/0 | 0/3 |
| put the brown cylindrical object on the tray | 2 | 1.00 | 0/0 | 0/2 |
| put the stuffed animal in the tray | 2 | 1.00 | 0/2 | 0/0 |
| put the sausage on the tray | 2 | 0.50 | 1/2 | 0/0 |
| put the bar on the tray | 1 | 1.00 | 0/0 | 0/1 |
| put the stuffed animal in the box | 1 | 1.00 | 0/1 | 0/0 |
| put the kiwi on the tray | 1 | 0.00 | 1/1 | 0/0 |
| put the moose on the tray | 1 | 1.00 | 0/1 | 0/0 |
| put the toy in the tray | 1 | 1.00 | 0/1 | 0/0 |
| put the brown plush toy on the tray | 1 | 1.00 | 0/1 | 0/0 |
| put the chocolate bar on the tray | 1 | 1.00 | 0/0 | 0/1 |
| put the chocolate on the tray | 1 | 1.00 | 0/0 | 0/1 |

## failure_type on judged failures (not scored)

| truly failed? | failure_type | n |
|---|---|---|
| yes | RETRY | 34 |
| yes | VOCAB | 28 |
| no (false fail) | RETRY | 1 |
| no (false fail) | VOCAB | 5 |

## Wrong verdicts

| id | instruction | expected | got | reason |
|---|---|---|---|---|
| 2026-08-06_23-31-04_01_t2_2 | put the stuffed animal on the tray | False | True | Pre-check: plush toy already on the tray. |
| 2026-08-19_03-51-37_01_t2_0 | put the banana on the tray | False | True | Pre-check: banana already on the tray. |
| 2026-08-19_07-36-23_02_t2_0 | put the banana on the tray | False | True | Pre-check: banana already on the tray. |
| 2026-08-20_06-49-37_01_t2_0 | put the banana on the tray | False | True | The banana is now on the tray as instructed. |
| 2026-08-20_06-49-37_01_t2_2 | put the banana on the tray | True | False | The arm reached the banana but the gripper closed on air and did not grasp it. |
| 2026-08-20_08-03-31_01_t2_0 | put the banana on the tray | False | True | Pre-check: banana already on the tray. |
| 2026-09-02_09-41-46_01_t2_0 | put the banana on the tray | False | True | Pre-check: banana already on the tray. |
| 2026-09-02_10-25-12_01_t1_1 | put the pouch on the tray | False | True | Pre-check: brown pouch already on the tray. |
| 2026-09-02_10-25-12_01_t3_0 | put the brown pouch on the tray | False | True | Pre-check: pouch already on the tray. |
| 2026-09-02_10-25-12_01_t3_1 | put the pouch on the tray | False | True | Pre-check: brown pouch already on the tray. |
| 2026-09-02_10-25-12_01_t3_2 | put the brown pouch on the tray | False | True | Pre-check: brown pouch already on the tray. |
| 2026-09-02_10-37-10_01_t1_1 | put the kiwi on the tray | True | False | The arm did not move toward the kiwi or the tray; it remained stationary, indicating no attempt to complete the task. |
| 2026-09-03_07-09-01_01_t2_0 | put the banana on the tray | False | True | Pre-check: banana already on the tray. |
| 2026-09-03_07-09-01_01_t2_1 | put the banana on the tray | False | True | Pre-check: banana already on the tray. |
| 2026-09-03_07-09-01_02_t2_1 | put the sausage on the tray | True | False | The arm did not move toward the sausage or the tray; it remained stationary, indicating no attempt to complete the task. |
| 2026-09-17_06-54-19_01_t1_2 | put the pouch on the tray | True | False | The arm did not move toward the pouch or the tray; it remained stationary, indicating no attempt to complete the sub-task. |
| 2026-09-17_06-54-19_01_t1_3 | put the pouch on the tray | True | False | The arm reached the pouch but did not grasp or move it to the tray. |
| 2026-09-17_06-54-19_01_t1_4 | put the pouch on the tray | True | False | The arm did not move toward the pouch or the tray; it remained stationary, indicating no attempt to complete the sub-task. |
| 2026-09-17_07-24-42_01_t2_0 | put the banana in the tray | False | True | Pre-check: banana already on the tray. |
