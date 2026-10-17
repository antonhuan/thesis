"""
Live (physical robot) eval of the FULL system on the preference prompts: the same
worklist, labels and scoring as vla_preference_eval_live.py (the raw-VLA baseline), but
each prompt goes through vlm_robot_orchestrator.run_high_level_task — VLM decomposition,
the VLA per sub-task, the live monitor, sub-task evaluation / retries / replans and the
final evaluation with its retry rounds.

The queue is the baseline's (preferences x --reps, --focus_keys x --focus_reps, prior x
--prior_reps, base x --base_reps, each prompt cycling the --layouts, shuffled with
--seed); the defaults are the settings of the baseline run in preference_eval_trials/, so
the queue is the same 57 (prompt, layout) items in the same order and every trial pairs
with a baseline trial. After each high-level task you label what the arm did over the
WHOLE task (every episode), exactly as in the baseline:
  * attempted  the objects the arm went for, in order (a real attempt: approached and
               tried to grasp it, whether or not the grasp held); n = none
  * placed     the objects on the tray at the end, in order
and the trial is scored with the baseline's score() (reach_admissible, intent_violated,
intent_complete, step_success, violated, complete, ... — see vla_preference_eval_live).
The VLM's own verdict is never the score: the final evaluation ignores order, so it
cannot grade the sequencing prompts.

Each trial writes <results_dir>/trial_NNN.json in the baseline's format, plus
  * system   the orchestrator's trace: outcome, the VLM's final verdict, every round
             (plan, replans, completed sub-tasks, final verdict) and every VLA episode
             (sub-task, instruction, episode dir, result, monitor outcome, judgement,
             motion), and vlm_agrees (VLM final verdict == labelled `complete`)
  * motion   in the baseline's shape, over the task's episodes: whole = all episodes,
             first = the first episode, after = the episodes after it (None if one) —
             one episode per sub-task, so "after the first placement" becomes "after the
             first sub-task episode". Per-episode metrics are under system.episodes.
  * episode  the first episode (dir + result), as in the baseline
The orchestrator's run folder (task.log, VLM frames, monitor records per task) is
<results_dir>/runs/run_<time>/NN/, and the VLA episodes (frames + actions.csv) are
under --vla_input_dir/<object>/<index>_<object>/. /stats prints the baseline tables plus
a SYSTEM table (VLM verdict, agreement with the labels, rounds, episodes, wall time).
Re-launching with the same --results_dir resumes, as in the baseline.

Usage (orchestrator flags, with the baseline's policy):
    python vlm_preference_eval_live.py \\
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=arm \\
        --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 342222071104, width: 640, height: 480, fps: 30}}" \\
        --task=dummy --server_address=127.0.0.1:8080 --policy_type=pi05 \\
        --pretrained_name_or_path=ant0nh/pi05_500_30k --policy_device=cuda \\
        --actions_per_chunk=50 --chunk_size_threshold=0.7
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.utils.import_utils import register_third_party_plugins
from vla_preference_baseline import CATEGORIES, OBJECTS
from vla_preference_eval_live import (CONTROL_CATEGORIES, PRIOR_KEY, PRIOR_SPECS, all_prompts,
                                      ask_layout, build_queue, describe_layout, done_counts,
                                      label_trial, layout_order, load_trials, motion_for,
                                      normalize_trial, parse_objects, print_stats, target_reps)
from vlm_monitor_live import TappedClient
from vlm_robot_orchestrator import (ABORT_KEY, InterjectionManager, OrchestratorConfig,
                                    TaskLogSink, VLMFrameSource, VLMPlanner, _Tee,
                                    run_high_level_task)


@dataclass
class SystemEvalConfig(OrchestratorConfig):
    # Queue (see vla_preference_eval_live.PreferenceEvalConfig); the defaults are the
    # baseline run's settings, so the queue matches preference_eval_trials/.
    reps: int = 2
    shuffle: bool = True
    seed: int = 0
    prior_reps: int = 6
    base_reps: int = 1
    focus_keys: list[str] = field(default_factory=lambda: ["qty one object", "qty two objects"])
    focus_reps: int = 5
    categories: list[str] = field(default_factory=lambda: list(CATEGORIES))
    slots: list[str] = field(default_factory=lambda: ["front", "back", "left"])
    layouts: list[str] = field(default_factory=lambda: ["ptb", "tbp", "bpt"])
    results_dir: str = "preference_eval_system_trials"
    vla_input_dir: str = "preference_eval_system_trials/episodes"


# ---------------------------------------------------------------------------
# Motion over a task's episodes, in the baseline's shape
# ---------------------------------------------------------------------------


def _combine(phases: list[dict]) -> dict | None:
    """One speed_metrics-shaped dict over several episodes: counts summed, maxima
    maxed, flags OR-ed."""
    if not phases:
        return None
    out = {"n_actions": sum(p["n_actions"] for p in phases)}
    for k in ("max_mean_speed", "max_peak_speed", "max_roll150_speed"):
        out[k] = max(p[k] for p in phases)
    for k in ("flag_velocity", "flag_rolling", "flag"):
        out[k] = any(p[k] for p in phases)
    return out


def task_motion(episodes: list[dict]) -> dict | None:
    """whole = every episode, first = the first episode, after = the rest (see the
    module docstring). Reads each episode's metrics from ep['motion']."""
    wholes = [e["motion"]["whole"] for e in episodes if e.get("motion")]
    if not wholes:
        return None
    return {"whole": _combine(wholes), "split_index": None, "split_t": None,
            "first": wholes[0], "after": _combine(wholes[1:]),
            "n_episodes": len(episodes)}


# ---------------------------------------------------------------------------
# System stats (on top of the baseline's tables)
# ---------------------------------------------------------------------------


def print_system_stats(trials: list[dict]):
    labelled = [t for t in trials if t.get("label_ok") and t.get("system")]
    if not labelled:
        return
    groups = {cat: [t for t in labelled if t["category"] == cat]
              for cat in ("prior", "base", *CATEGORIES, "custom")}
    groups["ALL preferences"] = [t for t in labelled if t["category"] not in CONTROL_CATEGORIES]
    groups["ALL"] = labelled

    def mean(xs):
        xs = [float(x) for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def pct(x, w):
        return f"{'-':>{w}}" if x is None else f"{x:{w}.0%}"

    def num(x, w):
        return f"{'-':>{w}}" if x is None else f"{x:{w}.1f}"

    print("\n  SYSTEM (VLM final verdict vs. labels; per high-level task)")
    print(f"  {'group':34} {'n':>3}  {'vlm ok':>7} {'complete':>8} {'agree':>6} "
          f"{'aborted':>7} {'rounds':>6} {'episodes':>8} {'mon stop':>8} {'wall s':>7}")
    for g, ts in groups.items():
        if not ts:
            continue
        s = [t["system"] for t in ts]
        print(f"  {g:34} {len(ts):3d}  {pct(mean(x['final_success'] for x in s), 7)} "
              f"{pct(mean(t['scores']['complete'] for t in ts), 8)} "
              f"{pct(mean(x['vlm_agrees'] for x in s), 6)} "
              f"{pct(mean(x['outcome'] == 'stopped' for x in s), 7)} "
              f"{num(mean(x['n_rounds'] for x in s), 6)} "
              f"{num(mean(x['n_episodes'] for x in s), 8)} "
              f"{num(mean(x['n_monitor_stops'] for x in s), 8)} "
              f"{num(mean(t.get('wall_s') for t in ts), 7)}")


def finish_system(out: dict):
    """(Re)compute the system fields that depend on the labels."""
    sysd = out.get("system")
    if sysd is not None:
        sysd["vlm_agrees"] = (None if sysd["final_success"] is None or not out.get("label_ok")
                              else sysd["final_success"] == out["scores"]["complete"])


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

HELP = f"""Commands:
  /run  (or Enter)   run the current prompt through the full system
                     ('{ABORT_KEY}'+Enter during the task aborts it)
  /list              all prompts with their admissible set and runs done
  /pref <key|#>      run this prompt next (key as in /list, e.g. 'omit banana', or its #)
  /task <text>       a custom prompt (set its admissible objects with /admissible)
  /admissible b,t    admissible objects for the custom prompt
  /layout <letters>  the layout for the next trial, objects per slot nearest-first
                     (e.g. /layout tbp); /layout alone shows the slots
  /skip              skip the current queue item
  /queue             show the remaining queue
  /relabel <trial#>  re-enter a saved trial's labels (e.g. an aborted / skipped one)
  /duration <s>      per sub-task episode     /home     /release (open, lift, home)
  /stats     /config     /help     /quit
  (label what the arm did over the WHOLE task — every episode; reset the scene after)"""


def main_repl(cfg: SystemEvalConfig):
    logging.basicConfig(level=logging.INFO)
    results = Path(cfg.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    prompts = all_prompts(cfg)
    keys = list(prompts)
    trials = load_trials(results)
    for t in trials:
        finish_system(t)
    trial_n = max((t.get("trial", 0) for t in trials), default=0)
    queue_ = build_queue(prompts, cfg, trials)
    custom = {"category": "custom", "prompt": None, "admissible": []}
    current: str | None = None  # a key forced with /pref or "custom"
    for lay in cfg.layouts:
        if layout_order(lay) is None:
            raise ValueError(f"layout '{lay}' is not a permutation of {OBJECTS}")
    layout_override: str | None = None  # set with /layout for the next trial
    last_layout: str | None = next((t.get("layout") for t in reversed(trials)
                                    if t.get("layout")), None)

    # The orchestrator's setup (vlm_robot_orchestrator.main): run folder, terminal
    # mirrored into each task's task.log, client, planner, frame source, interjection.
    run_dir = results / "runs" / f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.action_log_dir = str(run_dir)
    log_sink = TaskLogSink()
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, log_sink)
    sys.stderr = _Tee(_orig_stderr, log_sink)
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setStream(sys.stderr)
    logging.info(pformat(asdict(cfg)))
    if cfg.enable_vlm_monitor:
        cfg.log_vla_inputs = True  # the monitor's frame tap relies on the input capture
    client = TappedClient(cfg)
    if not client.connect_server():
        client.logger.error("Could not connect to policy server. Exiting.")
        client.stop()
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        return
    planner = VLMPlanner(cfg.vlm_model, temperature=cfg.vlm_temperature,
                         eval_fps=cfg.vlm_eval_fps,
                         two_pass=cfg.vlm_two_pass_decompose,
                         reask_uncovered=cfg.vlm_reask_uncovered,
                         filter_fixtures=cfg.vlm_filter_fixtures)
    frames = VLMFrameSource(client, camera_key=cfg.vlm_camera_key,
                            session_dir=run_dir, save_frames=cfg.save_frames)
    interjection = InterjectionManager(client)

    def current_item() -> tuple[str, dict, str] | None:
        """(key, spec, layout) of the next trial."""
        if current is not None:
            spec = custom if current == "custom" else prompts[current]
            lay = layout_override or last_layout or cfg.layouts[0]
            return current, spec, lay
        if not queue_:
            return None
        k, lay = queue_[0]
        return k, prompts[k], layout_override or lay

    def show_config():
        print(f"  prompts: {len(prompts)} (prior x{target_reps(PRIOR_KEY, PRIOR_SPECS[PRIOR_KEY], cfg)}, "
              f"base x{cfg.base_reps}, {', '.join(cfg.categories)} x{cfg.reps}, "
              f"{', '.join(cfg.focus_keys)} x{cfg.focus_reps})  shuffle={cfg.shuffle} "
              f"(seed {cfg.seed})\n  queue remaining: {len(queue_)}  saved trials: "
              f"{len(trials)}  duration={cfg.episode_duration}s per sub-task  "
              f"policy={cfg.pretrained_name_or_path}  vlm={cfg.vlm_model}\n  retries="
              f"{cfg.max_retries} replans={cfg.max_replans} final_retries="
              f"{cfg.max_final_retries} monitor={cfg.enable_vlm_monitor}  run dir {run_dir}")

    print("=" * 64 + "\nFULL SYSTEM PREFERENCE EVAL (prompt -> VLM -> VLA, monitored)\n"
          + HELP + "\n" + "=" * 64)
    show_config()
    client.go_home()
    try:
        while True:
            item = current_item()
            tag = (f"{item[0]}: '{item[1]['prompt']}' A_p={item[1]['admissible']} | layout "
                   f"{item[2]}" if item else "queue empty — /pref, /task or /quit")
            try:
                line = input(f"\n[{tag}] > ").strip()
            except EOFError:
                break
            cmd, _, arg = line.partition(" ")
            arg = arg.strip()
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/config":
                show_config()
            elif cmd == "/list":
                done = done_counts(trials)
                for i, (k, s) in enumerate(prompts.items()):
                    print(f"  {i:2d} {k:22} [{s['category']:11}] A_p={','.join(s['admissible']):18} "
                          f"done {done.get(k, 0)}/{target_reps(k, s, cfg)}  '{s['prompt']}'")
            elif cmd == "/pref" and arg:
                k = keys[int(arg)] if arg.isdigit() and int(arg) < len(keys) else arg
                if k in prompts:
                    current = k
                else:
                    print(f"  unknown prompt '{arg}' — see /list")
            elif cmd == "/task" and arg:
                custom["prompt"], current = arg, "custom"
                adm = ", ".join(custom["admissible"]) or "none — set them with /admissible"
                print(f"  custom prompt set; admissible objects: {adm}")
            elif cmd == "/admissible":
                objs = parse_objects(arg)
                if objs is None:
                    print("  objects as letters or names, e.g. /admissible b,t")
                else:
                    custom["admissible"] = objs
            elif cmd == "/skip":
                if current is not None:
                    current = None
                elif queue_:
                    queue_.pop(0)
            elif cmd == "/queue":
                print("  " + ("\n  ".join(f"{i:2d} {k:22} layout {lay}"
                                         for i, (k, lay) in enumerate(queue_)) or "(empty)"))
            elif cmd == "/layout":
                if not arg:
                    print(f"  slots nearest-first: {', '.join(cfg.slots)}; layouts: "
                          + "; ".join(f"{lay} ({describe_layout(layout_order(lay), cfg.slots)})"
                                      for lay in cfg.layouts))
                elif layout_order(arg):
                    layout_override = arg.lower()
                    print(f"  next trial: {describe_layout(layout_order(arg), cfg.slots)}")
                else:
                    print("  three letters, one per object, nearest slot first, e.g. /layout tbp")
            elif cmd == "/relabel" and arg.isdigit():
                path = results / f"trial_{int(arg):03d}.json"
                old = next((t for t in trials if t.get("trial") == int(arg)), None)
                if old is None or not path.exists():
                    print(f"  no saved trial {arg}")
                    continue
                print(f"  trial {arg}: {old['key']} '{old['prompt']}' A_p={old['admissible']} "
                      f"— currently attempted {old.get('attempted')}, placed {old.get('placed')}"
                      f"{'' if old.get('label_ok') else ' (unlabelled)'}")
                was_ok = bool(old.get("label_ok"))
                lay = ask_layout(old.get("layout"), cfg.slots)
                old["layout"], old["layout_order"] = lay, (layout_order(lay) if lay else None)
                if not label_trial(old):
                    # Cancelled: drop the in-memory edits, keep what is on disk.
                    restored = normalize_trial(json.loads(path.read_text()))
                    finish_system(restored)
                    trials[trials.index(old)] = restored
                    print("  relabel cancelled — trial unchanged")
                else:
                    finish_system(old)
                    old["relabelled"] = datetime.now().isoformat(timespec="seconds")
                    path.write_text(json.dumps(old, indent=2, default=str))
                    print(f"  saved {path}")
                    if not was_ok:  # now counts toward its reps
                        it = next((x for x in queue_ if x == (old["key"], lay)), None) \
                            or next((x for x in queue_ if x[0] == old["key"]), None)
                        if it:
                            queue_.remove(it)
            elif cmd == "/duration" and arg:
                cfg.episode_duration = float(arg)
            elif cmd == "/home":
                client.go_home()
            elif cmd == "/release":
                client.release_and_home()
            elif cmd == "/stats":
                print_stats(trials)
                print_system_stats(trials)
            elif cmd in ("", "/run"):
                if item is None or not item[1]["prompt"]:
                    print("  nothing to run — /pref <key> or /task <text>")
                    continue
                key, spec, layout = item
                order = layout_order(layout)
                ans = input(f"  SET LAYOUT {layout}: {describe_layout(order, cfg.slots)} "
                            "(nothing on the tray) — Enter to start, s to cancel: ")
                if ans.strip().lower() == "s":
                    continue
                trial_n += 1
                print(f"\n--- trial {trial_n}: {key} [{spec['category']}] '{spec['prompt']}' "
                      f"| A_p={spec['admissible']} | layout {layout} "
                      f"('{ABORT_KEY}'+Enter aborts) ---")
                t0 = time.time()
                interjection.start(abort_only=not cfg.enable_interjection)
                try:
                    trace = run_high_level_task(client, planner, frames, cfg, spec["prompt"],
                                                interjection, log_sink)
                finally:
                    interjection.stop()
                wall_s = round(time.time() - t0, 1)
                for ep in trace["episodes"]:
                    ep["motion"] = motion_for(ep["dir"])
                eps = trace["episodes"]
                trace.update(
                    n_rounds=len(trace["rounds"]), n_episodes=len(eps),
                    n_replans=sum(len(r["replans"]) for r in trace["rounds"]),
                    n_monitor_stops=sum(bool((e.get("monitor") or {}).get("stopped"))
                                        for e in eps),
                    vlm_agrees=None)
                print(f"  system: {trace['outcome']} | VLM final verdict "
                      f"{trace['final_success']} | {trace['n_rounds']} round(s), "
                      f"{len(eps)} episode(s), {trace['n_replans']} replan(s), "
                      f"{trace['n_monitor_stops']} monitor STOP(s) | {wall_s}s")
                for r in trace["rounds"]:
                    print(f"    round {r['round']}: planned {r['planned']} -> completed "
                          f"{r['completed']}")
                out = {"trial": trial_n, "time": datetime.now().isoformat(timespec="seconds"),
                       "key": key, "category": spec["category"], "prompt": spec["prompt"],
                       "admissible": list(spec["admissible"]),
                       "spec": {k: v for k, v in spec.items() if k not in ("category", "prompt")},
                       "layout": layout, "layout_order": order,
                       "layout_slots": list(cfg.slots)}
                # Motion is attached after labelling: the baseline's report line reads a
                # first-placement split, which a per-sub-task run does not have.
                labelled = label_trial(out)
                out["motion"] = task_motion(eps)
                for e in eps:
                    w = (e.get("motion") or {}).get("whole")
                    if w:
                        print(f"     motion '{e['instruction']}': "
                              f"{'OSCILLATION' if w['flag'] else 'no oscillation'} (mean "
                              f"{w['max_mean_speed']:.2f}, roll150 {w['max_roll150_speed']:.2f}, "
                              f"peak {w['max_peak_speed']:.0f})")
                out["system"] = trace
                finish_system(out)
                if trace["final_success"] is not None and labelled:
                    print(f"     VLM verdict {'AGREES' if trace['vlm_agrees'] else 'DISAGREES'} "
                          f"with the labels (complete={out['scores']['complete']})")
                note = input("Note (optional): ").strip()
                first = eps[0] if eps else None
                out.update(note=note,
                           episode=({"dir": first["dir"], **first["result"]} if first else None),
                           wall_s=wall_s,
                           config={k: getattr(cfg, k) for k in (
                               "pretrained_name_or_path", "policy_type", "actions_per_chunk",
                               "chunk_size_threshold", "episode_duration", "reps", "prior_reps",
                               "base_reps", "focus_keys", "focus_reps", "layouts", "slots", "seed",
                               "convergence_grace_period", "action_stall_timeout", "vlm_model",
                               "vlm_temperature", "evaluate_subtasks", "evaluate_final",
                               "max_retries", "max_replans", "max_final_retries",
                               "final_eval_before_after", "targeted_retry_decompose",
                               "subtask_done_precheck", "enable_vocab_refinement",
                               "enable_vlm_monitor")})
                path = results / f"trial_{trial_n:03d}.json"
                path.write_text(json.dumps(out, indent=2, default=str))
                trials.append(out)
                print(f"  saved {path} (task log: {trace['task_dir']}/task.log)")
                last_layout, layout_override = layout, None
                if labelled:
                    if current is not None:
                        current = None
                    elif queue_ and queue_[0][0] == key:
                        queue_.pop(0)
                print("  reset the scene before the next trial")
            else:
                print("Unknown command or missing argument. /help for commands.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        client.release_and_home()
        frames.stop()
        client.stop()
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        log_sink.close()
        print_stats(trials)
        print_system_stats(trials)


@draccus.wrap()
def main(cfg: SystemEvalConfig):
    main_repl(cfg)


if __name__ == "__main__":
    register_third_party_plugins()
    main()
