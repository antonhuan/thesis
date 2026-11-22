# VLM decompose test

- scene `2026-08-06_23-04-44`: `runs/run_2026-08-06_23-04-44/01/decompose_20260806_230850.png`
- scene `2026-08-06_23-31-04`: `runs/run_2026-08-06_23-31-04/01/decompose_20260806_233523.png`
- scene `2026-08-19_03-51-37`: `runs/run_2026-08-19_03-51-37/01/decompose_20260819_035832.png`
- scene `2026-08-19_04-10-27`: `runs/run_2026-08-19_04-10-27/01/decompose_20260819_041511.png`
- scene `2026-08-19_04-21-15`: `runs/run_2026-08-19_04-21-15/01/decompose_20260819_042508.png`
- scene `2026-08-19_04-44-51`: `runs/run_2026-08-19_04-44-51/01/decompose_20260819_044841.png`
- scene `2026-08-19_05-50-13`: `runs/run_2026-08-19_05-50-13/01/decompose_20260819_060122.png`
- scene `2026-08-19_06-31-10`: `runs/run_2026-08-19_06-31-10/01/decompose_20260819_063558.png`
- scene `2026-08-19_06-57-56`: `runs/run_2026-08-19_06-57-56/01/decompose_20260819_070128.png`
- scene `2026-08-19_07-17-03`: `runs/run_2026-08-19_07-17-03/01/decompose_20260819_072035.png`
- scene `2026-08-19_07-36-23`: `runs/run_2026-08-19_07-36-23/01/decompose_20260819_073953.png`
- scene `2026-08-20_06-23-01`: `runs/run_2026-08-20_06-23-01/01/decompose_20260820_062347.png`
- scene `2026-08-20_06-49-37`: `runs/run_2026-08-20_06-49-37/01/decompose_20260820_065026.png`
- scene `2026-08-20_07-17-48`: `runs/run_2026-08-20_07-17-48/01/decompose_20260820_071829.png`
- scene `2026-08-20_07-24-04`: `runs/run_2026-08-20_07-24-04/01/decompose_20260820_072449.png`
- scene `2026-08-20_08-03-31`: `runs/run_2026-08-20_08-03-31/01/decompose_20260820_080408.png`
- scene `2026-09-02_09-11-20`: `runs/run_2026-09-02_09-11-20/01/decompose_20260902_091153.png`
- scene `2026-09-02_09-41-46`: `runs/run_2026-09-02_09-41-46/01/decompose_20260902_094426.png`
- scene `2026-09-02_10-07-14`: `runs/run_2026-09-02_10-07-14/01/decompose_20260902_100750.png`
- scene `2026-09-02_10-25-12`: `runs/run_2026-09-02_10-25-12/01/decompose_20260902_102548.png`
- scene `2026-09-02_10-37-10`: `runs/run_2026-09-02_10-37-10/01/decompose_20260902_103754.png`
- scene `2026-09-03_06-00-35`: `runs/run_2026-09-03_06-00-35/01/decompose_20260903_060419.png`
- scene `2026-09-03_06-20-51`: `runs/run_2026-09-03_06-20-51/01/decompose_20260903_062125.png`
- scene `2026-09-03_07-09-01`: `runs/run_2026-09-03_07-09-01/01/decompose_20260903_070947.png`
- scene `2026-09-04_00-24-27`: `runs/run_2026-09-04_00-24-27/01/decompose_20260904_002507.png`
- scene `2026-09-09_07-03-56`: `runs/run_2026-09-09_07-03-56/01/decompose_20260909_070434.png`
- scene `2026-09-09_07-16-05`: `runs/run_2026-09-09_07-16-05/01/decompose_20260909_071646.png`
- scene `2026-09-10_02-36-03`: `runs/run_2026-09-10_02-36-03/01/decompose_20260910_023638.png`
- scene `2026-09-10_02-49-37`: `runs/run_2026-09-10_02-49-37/01/decompose_20260910_025038.png`
- scene `2026-09-10_03-02-18`: `runs/run_2026-09-10_03-02-18/01/decompose_20260910_030339.png`
- scene `2026-09-10_04-41-17`: `runs/run_2026-09-10_04-41-17/01/decompose_20260910_044211.png`
- scene `2026-09-10_04-49-31`: `runs/run_2026-09-10_04-49-31/01/decompose_20260910_045017.png`
- scene `2026-09-16_07-21-11`: `runs/run_2026-09-16_07-21-11/01/decompose_20260916_072627.png`
- scene `2026-09-16_08-18-24`: `runs/run_2026-09-16_08-18-24/01/decompose_20260916_081857.png`
- scene `2026-09-16_08-31-16`: `runs/run_2026-09-16_08-31-16/01/decompose_20260916_083157.png`
- scene `2026-09-16_09-04-56`: `runs/run_2026-09-16_09-04-56/01/decompose_20260916_090530.png`
- scene `2026-09-17_06-41-16`: `runs/run_2026-09-17_06-41-16/01/decompose_20260917_064341.png`
- scene `2026-09-17_06-54-19`: `runs/run_2026-09-17_06-54-19/01/decompose_20260917_065444.png`
- scene `2026-09-17_07-24-42`: `runs/run_2026-09-17_07-24-42/01/decompose_20260917_072520.png`
- model: `Qwen/Qwen3-VL-8B-Instruct` (temp 0.1)
- trials per prompt per scene: 1

| category | key | prompt | identified | object_set | order | count | destination | first pass | retried | correct | manual | plans |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | put everything on the tray | 0.72 | 0.72 | 1.00 | 0.51 | 0.97 | 0.51 | 0/39 | 0.51 | 0/39 | banana → toy → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch → None → None; banana → toy → pouch; banana → toy; banana → toy; banana → toy; banana → toy; toy → banana → pouch → None → None; toy → banana → pouch; toy → pouch → banana → None; toy → banana → pouch → None → None; toy → banana → pouch → None; toy → banana → pouch; toy → banana; toy → banana → None → None; toy → banana → pouch → None → None → None; toy → banana → pouch; toy → banana → None; toy → banana; toy → banana → None; toy → banana → pouch; banana → toy → pouch; toy → pouch → banana; toy → banana → pouch → None → None; toy → banana → pouch; toy → banana → pouch; banana → toy → pouch; banana → toy; toy → pouch → banana; banana → toy → pouch → None; toy → banana → pouch; toy → pouch → banana; banana → toy → None → None; banana → toy → pouch; toy → banana → pouch; toy → banana → pouch |
| omission | omit banana | put everything on the tray except the banana | 0.72 | 0.72 | 1.00 | 0.59 | 0.97 | 0.59 | 0/39 | 0.59 | 0/39 | toy → pouch; toy → pouch; toy → None → None → pouch; toy → pouch; toy → pouch; toy → pouch; toy; toy; toy; toy; toy → pouch; toy → pouch; toy → pouch → None; toy → pouch; toy → pouch; toy → pouch; toy; toy → None → None → None; toy → pouch; toy → pouch; toy → None; toy → None → None; toy → None; toy → pouch; toy → pouch; toy → pouch → None → None → None; toy → pouch; toy → pouch; toy → pouch; toy → pouch; toy; toy → pouch → None → None; toy → pouch → None; toy → pouch; toy → pouch; toy → None → None; toy → pouch; toy → pouch; toy → pouch |
| omission | omit toy | put everything on the tray except the toy | 0.74 | 0.74 | 1.00 | 0.64 | 1.00 | 0.64 | 0/39 | 0.64 | 0/39 | banana → pouch; banana → pouch; banana → pouch; banana → pouch; banana → pouch → None → None; banana → pouch; banana; banana; banana; banana; banana → pouch; banana → pouch; pouch → banana → None; banana → pouch; banana → pouch; banana → pouch; banana; banana → None → None; banana → pouch; banana → pouch; banana → None; banana; banana → None; banana → pouch; banana → pouch; pouch → banana; banana → pouch; banana → pouch; banana → pouch; banana → pouch; banana → pouch; pouch → banana → None → None; banana → pouch → None; banana → pouch; pouch → banana; banana → None → None; banana → pouch; banana → pouch; banana → pouch |
| omission | omit pouch | put everything on the tray except the pouch | 0.74 | 0.59 | 1.00 | 0.49 | 0.97 | 0.31 | 0/39 | 0.31 | 0/39 | banana → toy; toy → banana; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch → None → None; banana → toy; banana → toy; banana → toy; banana → toy; banana → toy; toy → banana → pouch; toy → banana → pouch; toy → pouch → banana → None; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana; toy → banana → None → None → None; toy → banana; toy → banana; toy → banana → None; toy → banana; toy → banana → None; toy → banana → pouch; banana → toy; toy → banana; toy → banana → pouch; toy → banana → pouch; toy → banana; banana → toy; banana → toy → pouch; toy → banana → pouch → None; banana → toy → None; toy → banana → pouch; toy → banana; banana → toy; banana → toy → pouch; toy → banana; toy → banana |
| omission | omit banana+toy | put everything on the tray except the banana and the toy | 0.72 | 0.72 | 1.00 | 0.59 | 0.79 | 0.59 | 0/39 | 0.59 | 0/39 | pouch; pouch; None → None → pouch; pouch; pouch; pouch; -; -; -; -; pouch; pouch; pouch → None; pouch; pouch → None; pouch; -; None → None; pouch; pouch; None; -; None; pouch; pouch; pouch → None; pouch; pouch; pouch; pouch; -; pouch; pouch → None; pouch; pouch; -; pouch; pouch; pouch |
| omission | omit toy+pouch | put everything on the tray but leave the toy and the pouch | 0.69 | 0.97 | 1.00 | 0.77 | 1.00 | 0.56 | 0/39 | 0.56 | 0/39 | banana; banana; banana → None → None; banana; banana; banana; banana; banana; banana; banana; banana; banana; banana → None; banana; banana → None → None; banana; banana; banana → None → None → None; banana; banana; banana → None; banana; banana → None; banana; banana; banana; banana; banana → pouch; banana; banana; banana; banana → None; banana → None; banana; banana; banana; banana; banana; banana |
| omission | omit don't touch | put everything on the tray, don't touch the pouch | 0.72 | 0.62 | 1.00 | 0.44 | 0.97 | 0.26 | 0/39 | 0.26 | 0/39 | banana → toy; toy → banana; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch → None → None; banana → toy; banana → toy; banana → toy; banana → toy; banana → toy; toy → banana → pouch; toy → banana → pouch; toy → pouch → banana → None; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana; toy → banana → None → None; toy → banana → None → None → None; toy → banana; toy → banana → None; toy → banana; toy → banana → None; toy → banana → pouch; banana → toy; toy → banana → None; toy → banana → pouch; toy → banana → pouch; toy → banana; banana → toy; banana → toy; toy → banana → pouch → None; banana → toy → None; toy → banana → pouch; toy → banana; banana → toy → None → None; banana → toy → pouch; toy → banana; toy → banana |
| sequencing | seq banana>toy>pouch | put the banana on the tray first, then the toy, then the pouch | 0.69 | 0.67 | 0.67 | 0.67 | 0.97 | 0.67 | 0/39 | 0.67 | 0/39 | banana → toy → pouch; banana → toy → pouch; banana → toy; banana → toy → pouch; banana → toy → pouch; banana → toy → pouch; banana → toy; banana → toy; banana → toy; banana → toy; banana → toy → pouch; banana → toy → pouch; banana → toy → pouch; banana → toy → pouch; banana → toy → None; banana → toy → pouch; banana → toy; banana → toy; banana → toy → pouch; banana → toy → pouch; banana → toy; banana → toy; banana → toy; banana → toy → pouch; banana → toy → pouch; banana → toy → pouch; banana → toy → pouch; banana → toy → pouch; banana → toy → pouch; banana → toy → pouch; banana → toy; banana → toy → pouch; banana → toy → pouch; banana → toy → pouch; banana → toy → pouch; banana → toy; banana → toy → pouch; banana → toy → pouch; banana → toy → pouch |
| sequencing | seq pouch>banana>toy | put the pouch on the tray first, then the banana, then the toy | 0.72 | 0.72 | 0.72 | 0.72 | 0.97 | 0.72 | 0/39 | 0.72 | 0/39 | pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; banana → toy; banana → toy; banana → toy; banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; banana → toy; None → banana → toy; pouch → banana → toy; pouch → banana → toy; None → banana → toy; banana → toy; banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy; banana → toy; pouch → banana → toy; pouch → banana → toy; pouch → banana → toy |
| sequencing | seq toy first | put everything on the tray, starting with the toy | 0.72 | 0.72 | 0.72 | 0.62 | 0.97 | 0.62 | 0/39 | 0.62 | 0/39 | toy → pouch → banana; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → pouch → banana; toy → banana; toy → banana; toy → banana; toy → banana; toy → banana → pouch; toy → banana → pouch; toy → pouch → banana → None; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana; toy → banana → None → None; toy → banana → pouch; toy → banana → pouch; toy → banana → None; toy → banana; toy → banana → None; toy → banana → pouch; toy → pouch → banana; toy → pouch → banana → None → None → None; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → pouch → banana; toy → banana; toy → banana → pouch → None; toy → pouch → banana → None; toy → banana → pouch; toy → pouch → banana; toy → banana; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch |
| sequencing | seq pouch last | put everything on the tray, but the pouch goes last | 0.69 | 0.69 | 0.64 | 0.59 | 0.97 | 0.59 | 0/39 | 0.59 | 0/39 | banana → toy → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; banana → toy → pouch; banana → toy; banana → toy; banana → toy; banana → toy; toy → banana → pouch; toy → banana → pouch; toy → pouch → banana → None; toy → banana → pouch → None; toy → banana → None; toy → banana → pouch; toy → banana; toy → banana → None → None → None → None → None; toy → banana → pouch; toy → banana → pouch; toy → banana → None; toy → banana → None → None; toy → banana → None; toy → banana → pouch; banana → toy → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; banana → toy → pouch; banana → toy; toy → pouch → banana → None; banana → toy → None → pouch; toy → banana → pouch; toy → banana → pouch; banana → toy; banana → toy → pouch; toy → banana → pouch; toy → banana → pouch |
| sequencing | seq banana after toy | put everything on the tray, put the banana after the toy | 0.69 | 0.67 | 1.00 | 0.59 | 0.97 | 0.59 | 0/39 | 0.59 | 0/39 | toy → pouch → banana; toy → banana → pouch; toy → banana; toy → banana → pouch; toy → banana → pouch → None → None; toy → banana → pouch; toy → banana; toy → banana; toy → banana; toy → banana; toy → banana → pouch; toy → banana → pouch; toy → pouch → banana; toy → banana → pouch; toy → banana; toy → banana → pouch; toy → banana; toy → banana; toy → banana → pouch; toy → banana → pouch; toy → banana → None; toy → banana; toy → banana; toy → banana → pouch; toy → pouch → banana; toy → pouch → banana; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → pouch → banana; toy → banana; toy → banana → pouch → None; toy → banana → pouch → None; toy → banana → pouch; toy → pouch → banana; toy → banana; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch |
| conditional | cond fruit | put the fruit on the tray | 0.74 | 0.87 | 1.00 | 0.87 | 1.00 | 0.62 | 0/39 | 0.62 | 0/39 | banana; banana; banana; banana → pouch; banana; banana; banana; banana; banana; banana; banana; banana → pouch; banana; banana; banana; banana; banana; banana; banana; banana; banana; banana; banana; banana; banana; banana; banana → pouch; banana; banana; banana; banana → pouch; banana; banana; banana → pouch; banana; banana; banana; banana; banana |
| conditional | cond brown | put the brown objects on the tray | 0.72 | 0.46 | 1.00 | 0.56 | 0.97 | 0.46 | 0/39 | 0.46 | 0/39 | toy → pouch; toy → pouch; pouch; pouch; toy → pouch → banana; toy → pouch; toy; toy; toy; toy; pouch; toy → pouch; toy → pouch; toy → pouch; pouch; pouch; toy → banana; toy → banana; toy → pouch; toy → pouch; toy → None; toy → banana; toy → banana; toy → pouch; toy → pouch; toy → pouch → banana; pouch; toy → pouch; toy → pouch; toy → pouch; toy; toy → pouch → banana; toy → pouch; pouch; toy → pouch; toy; toy → pouch; toy → pouch; toy → pouch |
| conditional | cond not edible | put everything that is not edible on the tray | 0.72 | 0.69 | 1.00 | 0.26 | 1.00 | 0.26 | 0/39 | 0.26 | 0/39 | toy → pouch → None; toy → pouch; toy → pouch → None; toy → pouch → None → None; toy → pouch → None; toy → pouch → None; toy; toy; toy; toy; toy → pouch → None → None; toy → pouch → None → None; toy → pouch → None; toy → pouch → None → None; toy → pouch → None; toy → pouch → None → None; toy → None → None → None; toy → None → None → None → None → None; toy → pouch → None; toy → pouch → None → None → None → None; toy → None; toy → None → None; toy → None; toy → pouch; toy → pouch; toy → pouch → None; toy → pouch → None; toy; toy → pouch; toy → pouch; toy; toy → pouch → None; toy → pouch → None; toy → pouch; toy → pouch; toy; toy → pouch; toy → pouch; toy → pouch |
| conditional | cond soft | put only the soft, furry things on the tray | 0.72 | 0.82 | 1.00 | 0.82 | 0.97 | 0.54 | 0/39 | 0.54 | 0/39 | toy → pouch; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy → pouch; toy; toy; toy; toy → pouch; toy → pouch; toy; toy; toy; toy; toy → pouch; toy; toy; toy → pouch; toy → pouch |
| conditional | cond if banana | if there is a banana on the table, put only the banana on the tray; otherwise put everything on the tray | 0.72 | 0.00 | 1.00 | 0.00 | 0.97 | 0.00 | 0/39 | 0.00 | 0/39 | banana → toy → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch → None → None; banana → toy → pouch; banana → toy; banana → toy; banana → toy; banana → toy; toy → banana → pouch; toy → banana → pouch; toy → pouch → banana → None; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana; toy → banana → None → None; toy → banana → pouch; toy → banana → pouch; toy → banana → None; banana → toy; toy → banana → None; toy → banana → pouch; banana → toy → pouch; toy → pouch → banana; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; banana → toy → pouch; banana → toy; toy → pouch → banana; banana → toy → pouch → None; toy → banana → pouch; toy → pouch → banana; banana → toy; banana → toy → pouch; toy → banana → pouch; toy → banana → pouch |
| conditional | cond if apple | if there is an apple on the table, put the apple on the tray; otherwise put the toy on the tray | 0.74 | 1.00 | 1.00 | 1.00 | 0.97 | 0.74 | 0/39 | 0.74 | 0/39 | toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy; toy |
| quantity | qty one | put one object on the tray | 0.77 | 0.79 | 1.00 | 0.79 | 1.00 | 0.74 | 0/39 | 0.74 | 0/39 | banana; toy; toy; toy; toy; banana; banana → toy; banana → toy; banana → toy; banana → toy; toy; toy; toy; toy; toy; toy; toy → banana; toy; toy; toy; toy; toy → banana; toy; toy; banana; toy; toy; toy; toy; banana; banana → toy → pouch; toy; toy; toy; toy; banana → toy; banana; toy; toy |
| quantity | qty two | put two objects on the tray | 0.69 | 1.00 | 1.00 | 1.00 | 0.97 | 0.69 | 0/39 | 0.69 | 0/39 | banana → toy; toy → banana; toy → banana; toy → banana; toy → banana; banana → toy; banana → toy; banana → toy; banana → toy; banana → toy; toy → banana; toy → banana; toy → pouch; toy → banana; toy → banana; toy → banana; toy → banana; toy → banana; toy → banana; toy → banana; toy → banana; toy → banana; toy → banana; toy → banana; banana → toy; toy → pouch; toy → banana; toy → banana; toy → banana; banana → toy; banana → toy; toy → pouch; banana → toy; toy → banana; toy → pouch; banana → toy; banana → toy; toy → banana; toy → banana |
| quantity | qty three | put all three objects on the tray | 0.69 | 0.69 | 1.00 | 0.69 | 0.97 | 0.69 | 0/39 | 0.69 | 0/39 | banana → toy → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; banana → toy → pouch; banana → toy; banana → toy; banana → toy; banana → toy; toy → banana → pouch; toy → banana → pouch; toy → pouch → banana; toy → banana → pouch; toy → banana → None; toy → banana → pouch; toy → banana; toy → banana → None; toy → banana → pouch; toy → banana → pouch; toy → banana → None; toy → banana; toy → banana → None; toy → banana → pouch; banana → toy → pouch; toy → pouch → banana; toy → banana → pouch; toy → banana → pouch; toy → banana → pouch; banana → toy → pouch; banana → toy; toy → banana → pouch; banana → toy → pouch; toy → banana → pouch; toy → pouch → banana; banana → toy; banana → toy → pouch; toy → banana → pouch; toy → banana → pouch |
| quantity | qty one brown | put one of the brown objects on the tray | 0.72 | 0.77 | 1.00 | 0.77 | 0.97 | 0.59 | 0/39 | 0.59 | 0/39 | pouch; pouch; toy; pouch; pouch; toy → pouch; toy; toy; toy; toy; pouch; pouch; toy → pouch; pouch; pouch; pouch; toy → banana; toy → banana; pouch; pouch; toy; toy → banana; toy → banana; pouch; toy → pouch; pouch; pouch; pouch; toy; toy → pouch; toy; pouch; toy → pouch; pouch; toy; toy; pouch; toy; pouch |
| quantity | qty two not banana | put two objects on the tray, but not the banana | 0.72 | 0.69 | 1.00 | 0.69 | 1.00 | 0.69 | 0/39 | 0.69 | 0/39 | toy → pouch; toy → pouch; toy → pouch; toy → pouch; toy → pouch; toy → pouch; toy; toy; toy; toy; toy → pouch; toy → pouch; toy → pouch; toy → pouch; toy → None; toy → pouch; toy; toy; toy → pouch; toy → pouch; toy → None; toy; toy; toy → pouch; toy → pouch; toy → pouch; toy → pouch; toy → pouch; toy → pouch; toy → pouch; toy; toy → pouch; toy → pouch; toy → pouch; toy → pouch; toy; toy → pouch; toy → pouch; toy → pouch |

## Per scene x category (overall correct)

| scene | base | omission | sequencing | conditional | quantity | all |
|---|---|---|---|---|---|---|
| 2026-08-06_23-04-44 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.87 |
| 2026-08-06_23-31-04 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 0.96 |
| 2026-08-19_03-51-37 | 1.00 | 0.17 | 0.60 | 0.50 | 1.00 | 0.57 |
| 2026-08-19_04-10-27 | 1.00 | 0.67 | 1.00 | 0.33 | 1.00 | 0.74 |
| 2026-08-19_04-21-15 | 0.00 | 0.50 | 0.80 | 0.50 | 1.00 | 0.65 |
| 2026-08-19_04-44-51 | 1.00 | 1.00 | 1.00 | 0.67 | 0.80 | 0.87 |
| 2026-08-19_05-50-13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2026-08-19_06-31-10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2026-08-19_06-57-56 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2026-08-19_07-17-03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2026-08-19_07-36-23 | 0.00 | 0.67 | 1.00 | 0.50 | 1.00 | 0.74 |
| 2026-08-20_06-23-01 | 1.00 | 0.67 | 1.00 | 0.50 | 1.00 | 0.78 |
| 2026-08-20_06-49-37 | 0.00 | 0.00 | 0.60 | 0.67 | 0.80 | 0.48 |
| 2026-08-20_07-17-48 | 0.00 | 0.67 | 0.80 | 0.67 | 1.00 | 0.74 |
| 2026-08-20_07-24-04 | 0.00 | 0.33 | 0.40 | 0.33 | 0.40 | 0.35 |
| 2026-08-20_08-03-31 | 1.00 | 0.67 | 1.00 | 0.50 | 1.00 | 0.78 |
| 2026-09-02_09-11-20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2026-09-02_09-41-46 | 0.00 | 0.00 | 0.00 | 0.17 | 0.20 | 0.09 |
| 2026-09-02_10-07-14 | 0.00 | 0.83 | 1.00 | 0.67 | 1.00 | 0.83 |
| 2026-09-02_10-25-12 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.91 |
| 2026-09-02_10-37-10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2026-09-03_06-00-35 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2026-09-03_06-20-51 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2026-09-03_07-09-01 | 1.00 | 0.67 | 1.00 | 0.83 | 1.00 | 0.87 |
| 2026-09-04_00-24-27 | 1.00 | 1.00 | 1.00 | 0.67 | 0.80 | 0.87 |
| 2026-09-09_07-03-56 | 1.00 | 0.50 | 0.80 | 0.50 | 1.00 | 0.70 |
| 2026-09-09_07-16-05 | 0.00 | 0.67 | 1.00 | 0.33 | 1.00 | 0.70 |
| 2026-09-10_02-36-03 | 1.00 | 0.50 | 1.00 | 0.67 | 1.00 | 0.78 |
| 2026-09-10_02-49-37 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.91 |
| 2026-09-10_03-02-18 | 1.00 | 1.00 | 1.00 | 0.67 | 0.80 | 0.87 |
| 2026-09-10_04-41-17 | 0.00 | 0.17 | 0.00 | 0.17 | 0.00 | 0.09 |
| 2026-09-10_04-49-31 | 1.00 | 0.17 | 0.40 | 0.50 | 1.00 | 0.52 |
| 2026-09-16_07-21-11 | 0.00 | 0.00 | 0.40 | 0.67 | 0.80 | 0.43 |
| 2026-09-16_08-18-24 | 1.00 | 0.67 | 1.00 | 0.50 | 1.00 | 0.78 |
| 2026-09-16_08-31-16 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.91 |
| 2026-09-16_09-04-56 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2026-09-17_06-41-16 | 1.00 | 0.67 | 1.00 | 0.83 | 1.00 | 0.87 |
| 2026-09-17_06-54-19 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.91 |
| 2026-09-17_07-24-42 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.91 |
| **all scenes** | **0.51** | **0.49** | **0.64** | **0.44** | **0.68** | **0.55** |

## Vague object names (resolved by elimination, worth a manual look)

188 trial(s) named an object only vaguely; the plan was scored as if the name meant the one object left over.

- 2026-08-19_03-51-37: 'brown object' -> pouch, 'object' -> pouch
- 2026-08-19_04-10-27: 'brown object' -> pouch
- 2026-08-19_04-21-15: 'brown object' -> pouch
- 2026-08-19_07-36-23: 'brown object' -> pouch
- 2026-08-20_08-03-31: 'brown object' -> pouch
- 2026-09-02_09-41-46: 'black object' -> pouch
- 2026-09-03_07-09-01: 'cigar' -> pouch
- 2026-09-09_07-16-05: 'brown object' -> pouch
- 2026-09-10_02-36-03: 'chocolate bar' -> pouch
- 2026-09-16_08-18-24: 'brown object' -> pouch

## Failure attribution (auto labels)

403 auto-labelled failures, of which 51 are not real failures (the missed object was not needed). Shares below are of the 352 real failures.

| cause | group | trials | share | meaning |
|---|---|---|---|---|
| perception_missing | perception | 120 | 34% | missed a needed object; reasoning correct on what it saw |
| perception_extra | perception | 65 | 18% | planned a non-object (arm, wire, ...); reasoning otherwise correct |
| perception_missing_extra | perception | 34 | 10% | both of the above; reasoning correct on what it saw |
| grounding | grounding | 28 | 8% | object perceived but named vaguely, so the preference could not refer to it |
| reasoning | reasoning | 56 | 16% | full perception; wrong plan |
| mixed | mixed | 49 | 14% | perception error AND wrong plan on the objects it saw |
| not_a_failure | none | 51 | - | missed object was not needed; the plan is correct |

By group: perception 219 (62%), reasoning 56 (16%), grounding 28 (8%), mixed 49 (14%)

| category | perception_missing | perception_extra | perception_missing_extra | grounding | reasoning_reask | reasoning | mixed | parse_error | not_a_failure |
|---|---|---|---|---|---|---|---|---|---|
| base | 7 | 8 | 3 | 0 | 0 | 0 | 1 | 0 | 0 |
| omission | 20 | 31 | 11 | 17 | 0 | 9 | 9 | 0 | 22 |
| sequencing | 44 | 9 | 9 | 0 | 0 | 2 | 7 | 0 | 0 |
| conditional | 12 | 17 | 6 | 11 | 0 | 38 | 19 | 0 | 29 |
| quantity | 37 | 0 | 5 | 0 | 0 | 7 | 13 | 0 | 0 |

## Per category

- base: 0.51 (39 trials)
- omission: 0.49 (234 trials)
- sequencing: 0.64 (195 trials)
- conditional: 0.44 (234 trials)
- quantity: 0.68 (195 trials)
