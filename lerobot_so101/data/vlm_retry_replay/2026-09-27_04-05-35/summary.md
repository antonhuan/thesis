# Retry-round replay

- model: `Qwen/Qwen3-VL-8B-Instruct` (temp 0.1)

| case | prompt | completed in round 1 | survey: not at destination | old round 2 | targeted round 2 |
|---|---|---|---|---|---|
| 2026-08-19_04-21-15_07 | put everything on the tray except for the toy | - | pouch, banana | put the banana on the tray; put the pouch on the tray | put the pouch on the tray; put the banana on the tray |
| 2026-08-19_04-44-51_01 | put everything on the tray except for the banana | - | banana, brown pouch | put the plush toy on the tray | put the brown pouch on the tray |
| 2026-08-19_05-50-13_01 | put everything on the tray except for the banana | put the toy on the tray | banana, brown pouch | put the pouch on the tray | put the brown pouch on the tray |
| 2026-08-19_05-50-13_04 | put everything on the tray except for the banana | put the toy on the tray | banana, pouch | put the pouch on the tray | put the pouch on the tray |
| 2026-08-19_06-57-56_02 | put everything on the tray except for the banana | put the toy on the tray | banana, brown cylindrical object | put the cylinder on the tray | put the brown cylindrical object on the tray |
| 2026-08-19_07-17-03_01 | put everything on the tray except for the banana | put the toy on the tray | banana, brown cylindrical object | put the brown cylinder on the tray | put the brown cylindrical object on the tray |
| 2026-08-20_06-23-01_01 | put everything away | put the stuffed animal in the tray; put the banana in the tray | brown pouch | put the pouch on the tray | put the brown pouch in the tray |
| 2026-08-20_06-23-01_02 | put everything on the tray except for the banana | put the toy on the tray | banana, brown cylindrical object | put the cylinder on the tray | put the brown cylindrical object on the tray |
| 2026-08-20_06-23-01_03 | put everything on the tray except for hte pouch | put the stuffed animal on the tray; put the banana on the tray | brown pouch, banana | (empty) | (empty) |
| 2026-08-20_06-49-37_01 | put everything on the tray except for the pouch | put the stuffed animal on the tray; put the banana on the tray | brown pouch | put the table on the tray; put the robot arm on the tray; put the wires on the tray; put the black cables on the tray | (empty) |
| 2026-08-20_07-24-04_01 | put everything on the tray | put the stuffed animal on the tray; put the banana on the tray | brown pouch | put the pouch on the tray | put the brown pouch on the tray |
| 2026-08-20_07-24-04_03 | put everything on the tray except for the toy | put the banana on the tray; put the bag on the tray | - | put the wires on the tray | STOP (survey: nothing off a destination) |
| 2026-08-20_08-03-31_02 | put everything on the tray | put the banana on the tray | brown pouch | put the pouch on the tray | put the brown pouch on the tray |
| 2026-08-20_08-03-31_03 | put everything on the tray | put the banana on the tray; put the stuffed animal on the tray | brown pouch | put the object on the tray | put the brown pouch on the tray |
| 2026-09-02_09-11-20_02 | put everything away | put the stuffed animal in the box | banana, plush toy, brown pouch | put the banana on the tray; put the pouch on the tray | put the banana on the tray; put the brown pouch on the tray |
| 2026-09-02_09-41-46_03 | put the banana on the tray | put the banana on the tray | plush toy, banana, pouch | put the chocolate bar on the tray; put the plush toy on the tray | put the banana on the tray |
| 2026-09-02_09-41-46_04 | put everything on the tray | put the plush toy on the tray; put the banana on the tray | pouch | put the pouch on the tray | put the pouch on the tray |
| 2026-09-02_10-25-12_01 | put everything on the tray | put the plush toy on the tray; put the banana on the tray | brown pouch | put the pouch on the tray | put the brown pouch on the tray |
| 2026-09-02_10-37-10_01 | put everything on the tray | put the plush toy on the tray; put the banana on the tray | brown pouch | put the pouch on the tray | put the brown pouch on the tray |
| 2026-09-03_06-00-35_01 | put everything on the tray | put the plush toy on the tray; put the banana on the tray | brown pouch | put the pouch on the tray | put the brown pouch on the tray |
| 2026-09-03_07-09-01_01 | put everything on the tray | put the plush toy on the tray; put the brown pouch on the tray | banana, brown pouch | put the banana on the tray | put the banana on the tray; put the brown pouch on the tray |
| 2026-09-06_05-53-28_01 | put everything away | put the plush toy in the tray; put the pouch in the tray | - | (empty) | STOP (survey: nothing off a destination) |
| 2026-09-09_02-42-03_01 | put everything on the tray | put the plush toy on the tray; put the pouch on the tray | - | put the robot arm on the tray | STOP (survey: nothing off a destination) |
| 2026-09-10_02-49-37_01 | put everything on the tray | put the toy on the tray; put the banana on the tray; put the brown plush toy on the tray | brown pouch | put the pouch on the tray | put the brown pouch on the tray |
| 2026-09-10_03-02-18_01 | put everything on the tray except for the banana | put the plush toy on the tray | banana, brown pouch | put the pouch on the tray | put the brown pouch on the tray |
| 2026-09-10_04-41-17_01 | put the food on the tray | put the banana on the tray | chocolate bar, stuffed animal, banana | put the chocolate bar on the tray; put the stuffed animal on the tray | put the chocolate bar on the tray; put the stuffed animal on the tray |
| 2026-09-16_08-18-24_01 | put everything on the tray | put the plush toy on the tray; put the banana on the tray | banana, brown pouch | put the brown pouch on the tray | put the banana on the tray; put the brown pouch on the tray |
| 2026-09-17_06-54-19_01 | put everything on the tray except for the banana | put the toy on the tray | banana, pouch | put the pouch on the tray | put the pouch on the tray |
