# Preference-conditioned pick-and-place: VLA baseline vs. full VLM–VLA system

## Aim

A VLA policy fine-tuned on single-object instructions ("put the banana on the tray") is
given instructions that carry a *preference*: an exclusion, an order, a condition on the
objects, or a count. This experiment measures whether the VLA can follow such preferences
on its own. It then measures how much is gained by placing it inside the full system,
where a VLM decomposes the preference into the base sub-tasks the VLA was trained on,
monitors each grasp, evaluates every sub-task and the final scene, and retries.

Questions:
1. **Intent:** does the system go for the objects the preference allows, in the order it
   requires, and stop where it should?
2. **Execution:** does the final tray match the preference?
3. **Failure sources:** which component causes the remaining failures?
4. **Ablation:** does in-round replanning after a failed sub-task help?

## Method

### Setup
- **Robot:** SO-101 arm, a wrist camera, and a top-down RealSense camera. The VLA is
  `ant0nh/pi05_500_30k` in both conditions.
- **Scene:** a pink tray and three objects: a banana, a brown plush moose toy and a brown
  cylindrical pouch.
- **Layouts:** objects sit in three slots, ordered nearest-first (front, back, left).
  Three layouts (`ptb`, `tbp`, `bpt`) make each object the nearest once, so a pure
  "take the nearest object" policy can be separated from following the instruction.

### Prompt set and worklist
The baseline's prompt set (`vla_preference_baseline.PREFERENCE_PROMPTS`):

| Category | Prompts | Trials | Example |
|---|---|---|---|
| Omission | 6 | 12 | "put everything on the tray except the banana and the toy" |
| Sequencing | 5 | 10 | "put the toy on the tray first, then the pouch, then the banana" |
| Conditional | 6 | 12 | "put everything that is not edible on the tray" |
| Quantity | 4 | 14 | "put two objects on the tray" |
| Prior (control) | 1 | 6 | "put everything on the tray" |
| Base (control) | 3 | 3 | "put the pouch on the tray" |

That gives **57 trials**. Each prompt cycles through the layouts, and the "one object" and
"two objects" prompts get 5 runs each. The queue is shuffled with seed 0. The full system
ran the *same* 57 (prompt, layout) items as the baseline, so every system trial is paired
with a baseline trial.

### Conditions
- **Baseline:** the raw preference prompt is sent straight to the VLA, with no VLM and
  no monitor, for one 60 s episode (`vla_preference_eval_live.py`).
- **Full system:** `vlm_preference_eval_live.py`, which drives
  `vlm_robot_orchestrator.run_high_level_task`. The VLM is Qwen3-VL-8B (temperature 0.1).
  1. **Decompose:** a two-pass planner identifies the visible objects, filters out
     fixtures, then decomposes the prompt into "put the X on the tray" sub-tasks.
  2. **Execute:** one VLA episode per sub-task, up to 100 s.
  3. **Live monitor:** at the grasp, the arm is frozen; a two-pass reach check runs, and
     the wrist camera verifies the held object. A confirmed stop ends the episode; the
     same sub-task is retried once, and a second stop skips it.
  4. **Sub-task evaluation:** a 32-frame clip of the episode is judged by the VLM. If the
     arm did not move, a before/after pre-check runs first. One retry is allowed
     (`max_retries=1`).
  5. **Final evaluation:** a before/after scene survey, the instruction's requirements
     (objects on / off the tray, counts), and a comparison in code. If the verdict is
     incomplete, one retry round (`max_final_retries=1`) re-plans over the objects not
     yet on the tray, with a count cap for count prompts.
- **Replanning ablation:** in-round replanning after a sub-task fails all its attempts
  (`max_replans=1`) was on for the first pass through the worklist. The replan fired in
  11 trials. Those 11 (prompt, layout) items were re-run with replanning off
  (`max_replans=0`), which is the final configuration.
  - **Replan ON** = the first pass.
  - **Replan OFF** = the 46 first-pass trials that never replanned (equivalent to replan
    off) plus the 11 re-runs.

### Labels and metrics
After every trial the operator recorded two sequences over the whole task, all episodes
included:
- **attempted:** the objects the arm went for, in order (a real approach and grasp
  attempt, held or not);
- **placed:** the objects on the tray at the end, in order.

Scoring uses the baseline's `score()` unchanged; the VLM's own verdicts are never the score.
- **Intent** (attempted): `reach_admissible` (first attempt is allowed by the
  preference), `intent_complete` (the attempted sequence satisfies the whole preference),
  `intent_violated` (it attempts an excluded or unselected object, breaks the order, or
  exceeds the count).
- **Execution** (placed): `step_success` (first placed object is allowed), `complete`
  (the placed sequence satisfies the preference), `violated`.
- **Prior lift:** `reach_admissible` minus what the prior ("put everything") trials'
  first-attempt distribution predicts for the same allowed set, by proximity rank.

The paired comparison uses McNemar's exact test; intervals are 95% Wilson intervals.

## Results

### Headline (all 57 paired trials, controls included)

| Metric | Baseline | System, replan ON | System, replan OFF |
|---|---|---|---|
| First attempt admissible | 33 (58%) | 55 (96%) | 55 (96%) |
| Intent complete | 10 (18%) | 43 (75%) | **49 (86%)** |
| Intent violated | 34 (60%) | 13 (23%) | **8 (14%)** |
| First placement admissible | 27 (47%) | 53 (93%) | 54 (95%) |
| **Task complete** | 13 (23%) [14–35%] | 39 (68%) [56–79%] | **46 (81%) [69–89%]** |
| **Violated** | 26 (46%) [33–58%] | 13 (23%) [14–35%] | **8 (14%) [7–25%]** |

On the **48 preference trials alone** (the 9 prior and base controls excluded), task
complete is **10/48 (21%) → 31/48 (65%) → 38/48 (79%)** and violated
**26/48 (54%) → 12/48 (25%) → 7/48 (15%)**. These are the sums of the four preference rows
of the category table below. The headline table adds the controls: 3 → 8 → 8 complete.

Paired against the baseline (replan OFF):
- **Complete:** 35 trials only the system completed, 2 only the baseline completed,
  p ≈ 1e−8.
- **Violated:** 22 trials only the baseline violated, 4 only the system violated,
  p ≈ 5e−4.
- **Intent complete:** 40 vs 1, p ≈ 4e−11.

**Operator aborts.** Three replan-OFF trials were aborted by the operator. In both
"put the yellow object on the tray" trials (#54, #58) the planner selected all three
objects; the operator aborted after the banana was placed, which left a "complete"
tray. Scored on what the system planned, these two are failures. Counted that way,
replan OFF completes **44/57 (77%)** overall and **36/48 (75%)** of the preference trials.

### By category (complete: baseline → replan ON → replan OFF)

| Category | n | Complete | Violated | Intent complete (base → OFF) |
|---|---|---|---|---|
| Omission | 12 | 0 → 10 → 11 | 12 → 2 → 1 | 0 → 11 |
| Sequencing | 10 | 0 → 3 → 5 | 6 → 4 → 3 | 0 → 8 |
| Conditional | 12 | 2 → 7 → 10 | 6 → 5 → 2 | 2 → 9 |
| Quantity | 14 | 8 → 11 → 12 | 2 → 1 → 1 | 5 → 13 |
| Prior | 6 | 0 → 6 → 6 | 0 → 0 → 0 | 0 → 6 |
| Base | 3 | 3 → 2 → 2 | 0 → 1 → 1 | 3 → 2 |

### Did the VLA respond to the preference?
- **Baseline:** the first attempt was admissible in 24 of 48 preference trials (50%;
  58% over all 57 trials including controls), against
  53% predicted from the prior's proximity-rank distribution: a **lift of −3 points**.
  Its first attempt was the nearest object in 65% of trials. When the prompt named an
  object, the first attempt was that object in 79% of trials (keyword latching),
  including the excluded object on omission prompts. All 12 baseline omission trials
  were violations.
- **Full system:** 96% admissible first attempts, a **+32-point lift** over the same
  prior. Nearest-first dropped to 21%, so the choice of object no longer follows
  proximity.

### Cost

| | Baseline | Replan ON | Replan OFF |
|---|---|---|---|
| Wall time per trial (mean) | 71 s | 185 s | 155 s (median 144 s) |
| VLA episodes per trial (mean) | 1 | 3.25 | 2.68 |
| Trials with a retry round | — | 19 | 16 |
| Monitor stops | — | 18 | 13 |

Oscillation (the baseline's motion metric) was flagged in 2 of 153 system episodes,
and in about 4% of baseline preference trials.

### Replanning ablation
- **Replan ON:** the replan fired in 11 of 57 trials, each after a sub-task was judged
  failed on both attempts:
  - 4 times on the pouch, which was already on the tray (sub-task evaluator false fails);
  - 5 times on an object name the VLA does not respond to, or on an invented object
    ("cylinder", "case", "pen case", "black object");
  - 2 times after a genuine grasp failure.
- **It never clearly recovered a trial.** Six of the 11 ended violated:
  - on conditional prompts it added the unselected banana (#15, #22, #41);
  - it kept an invented or unknown name, which the VLA or the monitor then grounded to
    the wrong object (#29 "black object", #46 "case");
  - #47 broke the order.

  It also planned three objects for "one object" (#36), which ended incomplete.
- **With replanning off,** 9 of the 11 re-runs completed. Only 3 of the 11 re-runs
  reached a point where a replan would have fired. Of those, one changed (#47, violated
  → complete); #44 completed and #15 was violated in both conditions.
- The overall ON → OFF gain (39 → 46 complete) is therefore mostly variance from
  re-running the episodes, not a measured effect of replanning. The direct evidence is
  that the replan caused six violations and recovered nothing.

### Component diagnostics (replan OFF)
- **Sub-task evaluator:** 35 of 137 judged attempts were failed. The no-movement
  pre-check rescued 21 attempts: the arm did not move because the object was already on
  the tray, nearly always the pouch after a false fail on the previous attempt (the pouch
  placed onto an occupied tray). The monitor's
  grasp freeze takes up most of the clip, and the placement appears only in the last few
  frames, brown on brown. Pouch false fails cost a retry episode each.
- **Final evaluator:** its verdict agreed with the labels in 41 of 54 trials (76%).
  - **10 false "incomplete":** a banana hidden under the stack read as "missing", or the
    pouch on a stack. Each triggers an unnecessary retry round.
  - **3 false "complete",** all sequencing prompts (#12, #32, #52): the final eval
    ignores order by design.
- **Live monitor:** 13 stops in 6 trials.
  - Correct wrong-object catches: #13, #35, #49, and the banana stops in #39.
  - False stops: #39's pouch, where it could not match "pouch" with "pencil case", and
    #12, confused by the banana lying on the toy.
  - #45 was a wrong stop that happened to prevent a planned violation.
- **Retry rounds:** replaying the 19 saved round-2 frames offline, the live round-2
  plans were correct in 9 of 19 and broke the prompt in 6. When told the task was
  "judged incomplete", the planner planned the leftover objects even when the prompt
  excluded them.

### Remaining failures (replan OFF)

The table lists the **11 trials that were not complete** (8 of them violated), plus the two
aborted "yellow object" trials, which are labelled complete but count as failures under the
stricter scoring.

| Source | Trials | Violated |
|---|---|---|
| **Round-1 planner error** | | |
| — "put the banana on the tray after the toy" became one malformed sub-task ("put the banana after the moose toy on the tray"); only the banana was placed | #32, #52 | 2 |
| — two objects planned for "one of the brown objects" | #20 | 1 |
| — all three objects selected for "the yellow object" (operator aborted; labelled complete, not among the 11) | #54, #58 | (0) |
| **Final eval misread → the retry round planned an excluded / unselected object** | #14, #15, #40, #50 | 4 |
| **Grasp failure in a sequencing task; the object was placed later and the order broke** | #12 | 1 |
| **Incomplete, nothing violated** | | |
| — banana grasp failed, operator aborted | #11 | 0 |
| — monitor repeatedly stopped the pouch grasp ("pouch" vs "pencil case") | #39 | 0 |
| — round 2 planned nothing; the survey missed the toy | #53 | 0 |

In #15 the final eval's requirements decided the toy is not brown, and in #50 they did
not recognise the "pencil case" as the excluded pouch.

## Discussion

**The system's gain comes from decomposition, not from better motor control.** Both
conditions use the same VLA. On the single-object base instructions it was trained on,
the VLA alone succeeds (3/3). On preference prompts it ignores the preference (lift −3)
and follows proximity or keyword latching. Omission makes this clearest: every baseline
omission trial placed an excluded object, while the system completed 11 of 12.
Decomposing the preference into the trained base instructions turns a preference-following
problem into one the VLA was trained to solve.

**Where the system still fails, the cause is VLM perception, and the closed loop
amplifies it.** The sub-task evaluator's false fails, the final evaluator's false
"incomplete" verdicts, and the monitor's stack confusion all come from the same weakness:
objects stacked or lying on each other in the tray, especially brown on brown. Each error
alone is mostly harmless; the no-movement pre-check absorbs most sub-task false fails.
They become violations when a corrective step acts on them. The replan and the retry round
were both triggered by false verdicts, and both then re-planned without keeping the
prompt's constraint. Of the 8 replan-OFF violations, 4 are this chain: a final-eval
misread followed by a retry round that planned an object the prompt excluded.

**Round-1 planning is reliable but not complete.** The other remaining violations come
from the first decomposition: "after the toy" was copied into the sub-task text (2/2
trials of that prompt), "one of the brown objects" got two sub-tasks, and "the yellow
object" selected all three objects. These are language-understanding failures of the
planner, not perception failures. They concentrate on phrasings that differ from the
planner's examples, which suggests targeted prompt examples or a code-side check of the
sub-task format and count.

**Corrective loops need to be conservative.** Turning off in-round replanning removed a
source of violations without losing recoveries. The same holds for retry rounds: offline,
round-2 plans improved from 9/19 to 14/19 correct when the "judged incomplete" framing was
replaced with a neutral one ("these sub-tasks are done; plan the instruction against the
current scene; nothing left is a valid answer"). Adding a requirements guard and a count
cap gave 16/19. A fresh identify pass did *worse* (6/19): it brought back invented objects.
That retry-round change is not part of the measured system. It is the obvious next step,
and it targets 4 of the 8 remaining violations.

**Naming is a hidden interface between the VLM and the VLA.** The VLM calls the pouch
"case", "pencil case", "cylinder" or "brown tube". The VLA was trained on one vocabulary
and does not move for unfamiliar names. The planner's exclusion check misses "case" for
"except the pouch" (#45), and the final-eval survey renames objects between rounds (#50).
A single object list per task, shared by the planner, monitor and evaluator, would
remove a class of errors that never shows up as a perception failure.

**Limitations.**
- One scene and three objects. 57 trials, mostly one run per (prompt, layout), so the
  per-category numbers are small.
- One operator labelled every trial, from live observation.
- The replan-OFF set combines first-pass trials with 11 re-runs recorded later in the
  same session.
- Configuration settled during the run: trial #1 used `max_retries=3` and 16-frame clips,
  and 34 trials ran with replanning enabled but never triggered it.
- Two "complete" outcomes depend on operator aborts (#54, #58).
- The baseline gets one 60 s episode while the system gets several. The comparison is
  between the policy alone and the policy as deployed in the system, not at matched
  compute; the system takes about 2.2× the baseline's wall time.

**Next steps.**
1. Neutral retry-round framing with the requirements guard and count cap; it is
   validated offline (16/19) and targets the retry-round violations.
2. One object vocabulary per task, shared across the planner, monitor and evaluator.
3. A post-homing still for the sub-task evaluator, or removing frozen frames from its
   clip, to cut pouch false fails.
4. Check round-1 plans in code: one object per sub-task in the standard "put the X on the
   tray" form, and the number of sub-tasks within any count in the prompt (#20, #32, #52).
5. Guard vocabulary refinement against swapping in another object's name (#17, #34).

---
Data: `preference_eval_system_trials/` (final replan-OFF set; replan-ON originals in
`archive/`), baseline in `preference_eval_trials/`, round-2 replay in `_r2replay/`.
