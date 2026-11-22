# VLM decompose test

- scene `scene`: `runs/run_2026-09-17_07-24-42/01/decompose_20260917_072520.png`
- model: `Qwen/Qwen3-VL-8B-Instruct` (temp 0.1)
- trials per prompt per scene: 3

| category | key | prompt | identified | object_set | order | count | destination | first pass | retried | correct | manual | plans |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | put everything on the tray | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy → banana → pouch; toy → banana → pouch; toy → banana → pouch |
| omission | omit banana | put everything on the tray except the banana | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy → pouch; toy → pouch; toy → pouch |
| omission | omit toy | put everything on the tray except the toy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | banana → pouch; banana → pouch; banana → pouch |
| omission | omit pouch | put everything on the tray except the pouch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy → banana; toy → banana; toy → banana |
| omission | omit banana+toy | put everything on the tray except the banana and the toy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | pouch; pouch; pouch |
| omission | omit toy+pouch | put everything on the tray but leave the toy and the pouch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | banana; banana; banana |
| omission | omit don't touch | put everything on the tray, don't touch the pouch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy → banana; toy → banana; toy → banana |
| sequencing | seq banana>toy>pouch | put the banana on the tray first, then the toy, then the pouch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | banana → toy → pouch; banana → toy → pouch; banana → toy → pouch |
| sequencing | seq pouch>banana>toy | put the pouch on the tray first, then the banana, then the toy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | pouch → banana → toy; pouch → banana → toy; pouch → banana → toy |
| sequencing | seq toy first | put everything on the tray, starting with the toy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy → banana → pouch; toy → banana → pouch; toy → banana → pouch |
| sequencing | seq pouch last | put everything on the tray, but the pouch goes last | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy → banana → pouch; toy → banana → pouch; toy → banana → pouch |
| sequencing | seq banana after toy | put everything on the tray, put the banana after the toy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy → banana → pouch; toy → banana → pouch; toy → banana → pouch |
| conditional | cond fruit | put the fruit on the tray | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | banana; banana; banana |
| conditional | cond brown | put the brown objects on the tray | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy → pouch; toy → pouch; toy → pouch |
| conditional | cond not edible | put everything that is not edible on the tray | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy → pouch; toy → pouch; toy → pouch |
| conditional | cond soft | put only the soft, furry things on the tray | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0/3 | 0.00 | 0/3 | toy → pouch; toy → pouch; toy → pouch |
| conditional | cond if banana | if there is a banana on the table, put only the banana on the tray; otherwise put everything on the tray | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0/3 | 0.00 | 0/3 | toy → banana → pouch; toy → banana → pouch; toy → banana → pouch |
| conditional | cond if apple | if there is an apple on the table, put the apple on the tray; otherwise put the toy on the tray | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy; toy; toy |
| quantity | qty one | put one object on the tray | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | 3/3 | 0.00 | 0/3 | toy → banana → pouch; toy → banana → pouch; toy → banana → pouch |
| quantity | qty two | put two objects on the tray | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | 3/3 | 0.00 | 0/3 | toy → banana → pouch; toy → banana → pouch; toy → banana → pouch |
| quantity | qty three | put all three objects on the tray | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy → banana → pouch; toy → banana → pouch; toy → banana → pouch |
| quantity | qty one brown | put one of the brown objects on the tray | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | 3/3 | 0.00 | 0/3 | toy → banana → pouch; toy → banana → pouch; toy → banana → pouch |
| quantity | qty two not banana | put two objects on the tray, but not the banana | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 | 1.00 | 0/3 | toy → pouch; toy → pouch; toy → pouch |

## Per scene x category (overall correct)

| scene | base | omission | sequencing | conditional | quantity | all |
|---|---|---|---|---|---|---|
| scene | 1.00 | 1.00 | 1.00 | 0.67 | 0.40 | 0.78 |
| **all scenes** | **1.00** | **1.00** | **1.00** | **0.67** | **0.40** | **0.78** |

## Failure attribution (auto labels)

15 auto-labelled failures, of which 0 are not real failures (the missed object was not needed). Shares below are of the 15 real failures.

| cause | group | trials | share | meaning |
|---|---|---|---|---|
| reasoning_reask | reasoning | 9 | 60% | full perception; first plan right, the uncovered-object re-ask broke it |
| reasoning | reasoning | 6 | 40% | full perception; wrong plan |

By group: reasoning 15 (100%)

| category | perception_missing | perception_extra | perception_missing_extra | grounding | reasoning_reask | reasoning | mixed | parse_error | not_a_failure |
|---|---|---|---|---|---|---|---|---|---|
| conditional | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| quantity | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 |

## Per category

- base: 1.00 (3 trials)
- omission: 1.00 (18 trials)
- sequencing: 1.00 (15 trials)
- conditional: 0.67 (18 trials)
- quantity: 0.40 (15 trials)
