# VLM sub-task evaluation test

- attempts: `vlm_subtask_eval_attempts.json` — fixed subset of 200 (seed 0)
- model: `Qwen/Qwen3-VL-8B-Instruct` (temp 0.1); planner.evaluate on the saved clip, with the clip's fps and the monitor's stop reason, as the orchestrator calls it

false fail = object on the tray but judged failed (a needless retry); false pass = not on the tray but judged done (the task moves on without it).

## Overall

| verdict | n | correct | false fail | false pass |
|---|---|---|---|---|
| this run | 200 | 0.88 | 23/125 | 2/75 |
| logged at the time (older prompts/models) | 200 | 0.84 | 30/125 | 1/75 |

## By group

| group | n | correct | false fail | false pass |
|---|---|---|---|---|
| object moved this attempt | 175 | 0.95 | 7/100 | 2/75 |
| already on the tray, arm did not move | 25 | 0.36 | 16/25 | 0/0 |
| target: banana | 52 | 0.92 | 2/30 | 2/22 |
| target: toy | 81 | 0.90 | 8/68 | 0/13 |
| target: pouch | 67 | 0.81 | 13/27 | 0/40 |

## By sub-task text

| instruction | n | correct | false fail | false pass |
|---|---|---|---|---|
| put the banana on the tray | 46 | 0.91 | 2/25 | 2/21 |
| put the stuffed animal on the tray | 27 | 1.00 | 0/21 | 0/6 |
| put the toy on the tray | 27 | 0.74 | 7/24 | 0/3 |
| put the pouch on the tray | 25 | 0.64 | 9/17 | 0/8 |
| put the plush toy on the tray | 17 | 1.00 | 0/14 | 0/3 |
| put the brown pouch on the tray | 8 | 0.88 | 1/2 | 0/6 |
| put the roll on the tray | 5 | 1.00 | 0/1 | 0/4 |
| put the cylinder on the tray | 4 | 1.00 | 0/1 | 0/3 |
| put the bag on the tray | 4 | 0.75 | 1/2 | 0/2 |
| put the purse on the tray | 4 | 1.00 | 0/0 | 0/4 |
| put the cylindrical object on the tray | 4 | 1.00 | 0/0 | 0/4 |
| put the banana in the tray | 3 | 1.00 | 0/2 | 0/1 |
| put the banana in the box | 3 | 1.00 | 0/3 | 0/0 |
| put the plush toy in the tray | 3 | 1.00 | 0/2 | 0/1 |
| put the pouch in the tray | 3 | 1.00 | 0/2 | 0/1 |
| put the pencil case on the tray | 3 | 1.00 | 0/0 | 0/3 |
| put the brown cylindrical object on the tray | 2 | 1.00 | 0/0 | 0/2 |
| put the stuffed animal in the tray | 2 | 1.00 | 0/2 | 0/0 |
| put the sausage on the tray | 2 | 0.00 | 2/2 | 0/0 |
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
| yes | RETRY | 42 |
| yes | VOCAB | 31 |
| no (false fail) | RETRY | 7 |
| no (false fail) | VOCAB | 16 |

## Wrong verdicts

| id | instruction | expected | got | reason |
|---|---|---|---|---|
| 2026-08-19_04-10-27_01_t2_0 | put the banana on the tray | True | False | The arm reached the banana but the gripper closed on air and did not grasp it. |
| 2026-08-20_06-49-37_01_t2_0 | put the banana on the tray | False | True | The banana is now on the tray as instructed. |
| 2026-08-20_06-49-37_01_t2_2 | put the banana on the tray | True | False | The arm reached the banana but did not grasp it, and the banana remained on the table. |
| 2026-08-20_07-24-04_03_t2_1 | put the bag on the tray | True | False | The arm grasped the brown cylindrical object but failed to place it on the tray; it was dropped onto the table. |
| 2026-08-20_08-03-31_02_t1_3 | put the toy on the tray | True | False | The arm did not move at all, so the toy was not placed on the tray. |
| 2026-08-20_08-03-31_02_t1_4 | put the toy on the tray | True | False | The arm did not move at all, so the toy was not placed on the tray. |
| 2026-08-20_08-03-31_02_t2_0 | put the toy on the tray | True | False | The arm did not move toward the toy or the tray; it remained stationary, indicating no attempt to complete the sub-task. |
| 2026-08-20_08-03-31_02_t2_1 | put the toy on the tray | True | False | The arm did not move at all, so the toy was not placed on the tray. |
| 2026-09-02_09-41-46_02_t1_0 | put the toy on the tray | True | False | The arm did not move toward the toy or attempt to grasp it; it remained stationary. |
| 2026-09-02_10-37-10_01_t1_1 | put the kiwi on the tray | True | False | The arm did not move toward the kiwi or the tray; it remained stationary, indicating no attempt to complete the task. |
| 2026-09-03_07-09-01_01_t2_3 | put the brown pouch on the tray | True | False | The arm reached the brown pouch but did not grasp it, and the pouch remained on the table. |
| 2026-09-03_07-09-01_02_t2_0 | put the sausage on the tray | True | False | The arm reached for the sausage but the gripper closed on air and did not grasp it. |
| 2026-09-03_07-09-01_02_t2_1 | put the sausage on the tray | True | False | The arm did not move toward the sausage or the tray; it remained stationary, indicating no attempt to complete the task. |
| 2026-09-09_07-03-56_01_t2_0 | put the pouch on the tray | True | False | The arm reached the pouch but did not grasp it, and the pouch remained on the table. |
| 2026-09-09_07-03-56_01_t2_1 | put the pouch on the tray | True | False | The arm did not move toward the pouch or attempt to place it on the tray. |
| 2026-09-10_04-49-31_01_t2_1 | put the banana on the tray | False | True | The banana is already on the tray as instructed. |
| 2026-09-16_08-18-24_01_t1_2 | put the pouch on the tray | True | False | The arm did not move toward the pouch or the tray; it remained stationary, indicating no attempt to complete the sub-task. |
| 2026-09-16_08-18-24_01_t1_3 | put the pouch on the tray | True | False | The arm did not move at all, so the pouch was not placed on the tray. |
| 2026-09-16_08-18-24_01_t1_4 | put the pouch on the tray | True | False | The arm did not move toward the pouch or the tray; it remained stationary, indicating no attempt to complete the task. |
| 2026-09-17_06-41-16_01_t2_0 | put the pouch on the tray | True | False | The arm reached the pouch but the gripper closed on air and did not grasp it. |
| 2026-09-17_06-54-19_01_t1_2 | put the pouch on the tray | True | False | The arm did not move toward the pouch or the tray; it remained stationary, indicating no attempt to complete the sub-task. |
| 2026-09-17_06-54-19_01_t1_3 | put the pouch on the tray | True | False | The arm reached toward the pouch but did not grasp it or move it to the tray. |
| 2026-09-17_06-54-19_01_t1_4 | put the pouch on the tray | True | False | The arm did not move toward the pouch or the tray; it remained stationary, indicating no attempt to complete the sub-task. |
| 2026-09-17_06-54-19_01_t1_5 | put the toy on the tray | True | False | The arm reached the banana but the gripper closed on air and did not grasp it. |
| 2026-09-17_06-54-19_01_t1_6 | put the toy on the tray | True | False | The arm did not move toward the toy or the tray; it remained stationary, indicating no attempt to complete the sub-task. |
