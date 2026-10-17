"""
Live (physical robot) HUMAN-ORACLE eval on the preference prompts: you stand in for the
VLM. For each preference ("put everything on the tray except the banana") you read the
prompt, watch the scene and type the instructions the VLA gets, one episode at a time —
normally the base sub-tasks it was trained on ("put the toy on the tray", shortcut 't'),
deciding the next one after watching the last (re-issue a failed grasp, stop when the
preference is satisfied). It is the upper bound of the decomposition: a perfect planner,
monitor and evaluator, with the same VLA executing.

The queue is the 48 preference items of the baseline's queue (vla_preference_eval_live.py:
preferences x --reps, --focus_keys x --focus_reps, each prompt cycling the --layouts,
shuffled with --seed). The queue is built with the baseline's controls in it and then
filtered, so with the default settings every item has the same (prompt, layout) and
relative order as in preference_eval_trials/ and preference_eval_system_trials/. The
prior / base controls are not run. Per trial:
  * instruction loop   b / t / p = "put the <obj> on the tray", r = repeat the last
                       instruction, any other text is sent verbatim, d = done. Each
                       episode ends with open gripper -> lift away -> home (Enter during
                       an episode aborts it). Do not touch the scene until done.
  * labels             as in the baseline, over the WHOLE task (every episode):
                       attempted (objects the arm went for, in order) and placed (objects
                       on the tray at the end), scored with the baseline's score().
Each trial writes <results_dir>/trial_NNN.json in the baseline's format, plus
  * oracle   the episodes (instruction, base object or None for free text, episode dir,
             result, motion), n_episodes, n_repeats (instructions re-issued for an object
             already instructed) and only_base (every instruction was a base sub-task)
  * motion   in the system eval's shape: whole = all episodes, first = the first, after
             = the rest (None if one)
  * episode  the first episode (dir + result), as in the baseline
VLA episodes (frames + actions.csv) go under --vla_input_dir/<object>/<index>_<object>/.
/stats prints the baseline tables, an ORACLE table and a COMPARE table of the preference
trials in --compare_dirs next to this run. Re-launching with the same --results_dir
resumes, as in the baseline.

Usage (same robot / policy flags as the baseline and system evals):
    python vla_preference_oracle_live.py \\
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=arm \\
        --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 342222071104, width: 640, height: 480, fps: 30}}" \\
        --task=dummy --server_address=127.0.0.1:8080 --policy_type=pi05 \\
        --pretrained_name_or_path=ant0nh/pi05_500_30k --policy_device=cuda \\
        --actions_per_chunk=50 --chunk_size_threshold=0.7
"""

import json
import logging
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import draccus

from lerobot.utils.import_utils import register_third_party_plugins
from vla_preference_baseline import BASE_PROMPTS, CATEGORIES, OBJECTS
from vla_preference_eval_live import (CONTROL_CATEGORIES, PreferenceEvalClient,
                                      PreferenceEvalConfig, all_prompts, ask_layout,
                                      build_queue, describe_layout, done_counts, label_trial,
                                      layout_order, load_trials, motion_for, normalize_trial,
                                      parse_objects, print_stats, target_reps)


@dataclass
class OracleEvalConfig(PreferenceEvalConfig):
    # Queue: the baseline run's settings (see vlm_preference_eval_live.SystemEvalConfig),
    # so the filtered queue is the baseline's 48 preference items. prior_reps / base_reps
    # only shape that queue — the controls are never run here.
    reps: int = 2
    prior_reps: int = 6
    base_reps: int = 1
    episode_duration: float = 60       # per instruction, as the baseline
    results_dir: str = "preference_eval_oracle_trials"
    vla_input_dir: str = "preference_eval_oracle_trials/episodes"
    # Runs shown next to this one in /stats (COMPARE table); missing dirs are skipped.
    compare_dirs: list[str] = field(default_factory=lambda: [
        "preference_eval_trials", "preference_eval_system_trials"])


def preference_queue(cfg: OracleEvalConfig, trials: list[dict]) -> list[tuple[str, str]]:
    """The baseline's queue (controls included, so layouts and order match) minus the
    controls and the labelled runs already saved."""
    full = all_prompts(cfg)
    return [(k, lay) for k, lay in build_queue(full, cfg, trials)
            if full[k]["category"] not in CONTROL_CATEGORIES]


# ---------------------------------------------------------------------------
# Motion over a trial's episodes (as vlm_preference_eval_live.task_motion; copied so
# this script does not import the VLM stack)
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
    wholes = [e["motion"]["whole"] for e in episodes if e.get("motion")]
    if not wholes:
        return None
    return {"whole": _combine(wholes), "split_index": None, "split_t": None,
            "first": wholes[0], "after": _combine(wholes[1:]),
            "n_episodes": len(episodes)}


# ---------------------------------------------------------------------------
# Stats (on top of the baseline's tables)
# ---------------------------------------------------------------------------


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pct(x, w):
    return f"{'-':>{w}}" if x is None else f"{x:{w}.0%}"


def _num(x, w):
    return f"{'-':>{w}}" if x is None else f"{x:{w}.1f}"


def _pref_groups(trials: list[dict]) -> "OrderedDict[str, list[dict]]":
    labelled = [t for t in trials if t.get("label_ok")
                and t["category"] not in CONTROL_CATEGORIES]
    groups = OrderedDict((cat, [t for t in labelled if t["category"] == cat])
                         for cat in CATEGORIES)
    groups["ALL preferences"] = labelled
    return groups


def print_oracle_stats(trials: list[dict]):
    groups = _pref_groups([t for t in trials if t.get("oracle")])
    if not groups["ALL preferences"]:
        return
    print("\n  ORACLE (human-prompted episodes per preference task)")
    print(f"  {'group':34} {'n':>3}  {'episodes':>8} {'repeats':>7} {'only base':>9} "
          f"{'complete':>8} {'violated':>8} {'wall s':>7}")
    for g, ts in groups.items():
        if not ts:
            continue
        o = [t["oracle"] for t in ts]
        print(f"  {g:34} {len(ts):3d}  {_num(_mean(x['n_episodes'] for x in o), 8)} "
              f"{_num(_mean(x['n_repeats'] for x in o), 7)} "
              f"{_pct(_mean(x['only_base'] for x in o), 9)} "
              f"{_pct(_mean(t['scores']['complete'] for t in ts), 8)} "
              f"{_pct(_mean(t['scores']['violated'] for t in ts), 8)} "
              f"{_num(_mean(t.get('wall_s') for t in ts), 7)}")


def print_comparison(trials: list[dict], compare_dirs: list[str]):
    """Preference trials of each run in compare_dirs and of this one, per category:
    intent complete (attempted) / complete / violated (placed)."""
    runs = [(d, load_trials(Path(d))) for d in compare_dirs if Path(d).is_dir()]
    runs.append(("oracle (this run)", trials))
    print("\n  COMPARE (preference trials: intent complete / complete / violated)")
    print(f"  {'group':18}" + "".join(f" {Path(name).name[:26]:>26}" for name, _ in runs))
    per_run = [_pref_groups(ts) for _, ts in runs]
    for g in per_run[-1]:
        cells = []
        for groups in per_run:
            ts = groups[g]
            cells.append(f"n={len(ts):<2} " + " / ".join(
                _pct(_mean(t["scores"][k] for t in ts), 4)
                for k in ("intent_complete", "complete", "violated")) if ts else "-")
        print(f"  {g:18}" + "".join(f" {c:>26}" for c in cells))


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

LETTERS = ", ".join(f"{o[0]} = '{BASE_PROMPTS[o]}'" for o in OBJECTS)

HELP = f"""Commands:
  /run  (or Enter)   start the current preference task, then prompt the VLA yourself:
                       {LETTERS}
                       r = repeat the last instruction, other text = sent verbatim,
                       d = done (then label the whole task)
                     (Enter during an episode aborts it)
  /list              the preference prompts with their admissible set and runs done
  /pref <key|#>      run this prompt next (key as in /list, e.g. 'omit banana', or its #)
  /layout <letters>  the layout for the next trial, objects per slot nearest-first
                     (e.g. /layout tbp); /layout alone shows the slots
  /skip              skip the current queue item
  /queue             show the remaining queue
  /relabel <trial#>  re-enter a saved trial's labels (e.g. an aborted / skipped one)
  /duration <s>      per instruction episode     /home     /release (open, lift, home)
  /stats     /config     /help     /quit
  (every episode ends with: open gripper -> lift away -> home; reset the scene after
   the task, not between its episodes)"""


def ask_instruction(last: str | None) -> tuple[str, str | None] | None:
    """(instruction, base object or None) for the next episode; None when done."""
    while True:
        ans = input(f"  instruction [{'/'.join(o[0] for o in OBJECTS)}, "
                    f"{'r = repeat, ' if last else ''}text, d = done]: ").strip()
        low = ans.lower()
        if low in ("d", "done", "/done"):
            return None
        if low == "r" and last:
            ans = last
        elif len(low) == 1:
            obj = parse_objects(low)
            if obj:
                ans = BASE_PROMPTS[obj[0]]
            else:
                print(f"  {LETTERS}, r, d, or a full instruction")
                continue
        if not ans:
            continue
        base = next((o for o, p in BASE_PROMPTS.items() if p == ans.lower()), None)
        return ans, base


def run_oracle_task(client: PreferenceEvalClient, cfg: OracleEvalConfig) -> list[dict]:
    """Prompt the VLA episode by episode until the oracle is done; the episode records."""
    episodes: list[dict] = []
    last: str | None = None
    while True:
        done = ", ".join(e["base"] or repr(e["instruction"]) for e in episodes)
        print(f"  episodes so far: {done or 'none'}")
        nxt = ask_instruction(last)
        if nxt is None:
            return episodes
        instruction, base = nxt
        print(f"  -> episode {len(episodes) + 1}: '{instruction}' (Enter aborts)")
        t0 = time.time()
        result = client.run_episode(instruction, enable_abort_listener=True)
        episode_dir = str(client._vla_input_dir)
        client.release_and_home()
        episodes.append({"instruction": instruction, "base": base, "dir": episode_dir,
                         "result": asdict(result), "wall_s": round(time.time() - t0, 1),
                         "motion": motion_for(episode_dir)})
        last = instruction


def main_repl(cfg: OracleEvalConfig):
    logging.basicConfig(level=logging.INFO)
    results = Path(cfg.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    prompts = OrderedDict((k, s) for k, s in all_prompts(cfg).items()
                          if s["category"] not in CONTROL_CATEGORIES)
    keys = list(prompts)
    trials = load_trials(results)
    trial_n = max((t.get("trial", 0) for t in trials), default=0)
    queue_ = preference_queue(cfg, trials)
    current: str | None = None  # a key forced with /pref
    for lay in cfg.layouts:
        if layout_order(lay) is None:
            raise ValueError(f"layout '{lay}' is not a permutation of {OBJECTS}")
    layout_override: str | None = None  # set with /layout for the next trial
    last_layout: str | None = next((t.get("layout") for t in reversed(trials)
                                    if t.get("layout")), None)

    client = PreferenceEvalClient(cfg)
    client.prompt_tag = None  # episode dirs by object, as in the system eval
    if not client.connect_server():
        client.logger.error("Could not connect to policy server. Exiting.")
        client.stop()
        return

    def current_item() -> tuple[str, dict, str] | None:
        """(key, spec, layout) of the next trial."""
        if current is not None:
            lay = layout_override or last_layout or cfg.layouts[0]
            return current, prompts[current], lay
        if not queue_:
            return None
        k, lay = queue_[0]
        return k, prompts[k], layout_override or lay

    def show_config():
        print(f"  preference prompts: {len(prompts)} ({', '.join(cfg.categories)} x{cfg.reps}, "
              f"{', '.join(cfg.focus_keys)} x{cfg.focus_reps})  shuffle={cfg.shuffle} "
              f"(seed {cfg.seed})\n  queue remaining: {len(queue_)}  saved trials: "
              f"{len(trials)}  duration={cfg.episode_duration}s per instruction  "
              f"policy={cfg.pretrained_name_or_path}")

    print("=" * 64 + "\nHUMAN-ORACLE PREFERENCE EVAL (you prompt the VLA, no VLM)\n"
          + HELP + "\n" + "=" * 64)
    show_config()
    client.go_home()
    try:
        while True:
            item = current_item()
            tag = (f"{item[0]}: '{item[1]['prompt']}' A_p={item[1]['admissible']} | layout "
                   f"{item[2]}" if item else "queue empty — /pref or /quit")
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
                instr = [e["instruction"] for e in (old.get("oracle") or {}).get("episodes", [])]
                print(f"  trial {arg}: {old['key']} '{old['prompt']}' A_p={old['admissible']} "
                      f"— instructions {instr}, currently attempted {old.get('attempted')}, "
                      f"placed {old.get('placed')}"
                      f"{'' if old.get('label_ok') else ' (unlabelled)'}")
                was_ok = bool(old.get("label_ok"))
                lay = ask_layout(old.get("layout"), cfg.slots)
                old["layout"], old["layout_order"] = lay, (layout_order(lay) if lay else None)
                if not label_trial(old):
                    # Cancelled: drop the in-memory edits, keep what is on disk.
                    trials[trials.index(old)] = normalize_trial(json.loads(path.read_text()))
                    print("  relabel cancelled — trial unchanged")
                else:
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
                print_oracle_stats(trials)
                print_comparison(trials, cfg.compare_dirs)
            elif cmd in ("", "/run"):
                if item is None:
                    print("  nothing to run — /pref <key>")
                    continue
                key, spec, layout = item
                order = layout_order(layout)
                ans = input(f"  SET LAYOUT {layout}: {describe_layout(order, cfg.slots)} "
                            "(nothing on the tray) — Enter to start, s to cancel: ")
                if ans.strip().lower() == "s":
                    continue
                trial_n += 1
                print(f"\n--- oracle trial {trial_n}: {key} [{spec['category']}] "
                      f"'{spec['prompt']}' | A_p={spec['admissible']} | layout {layout} ---")
                t0 = time.time()
                eps = run_oracle_task(client, cfg)
                wall_s = round(time.time() - t0, 1)
                bases = [e["base"] for e in eps]
                oracle = {"episodes": eps, "instructions": [e["instruction"] for e in eps],
                          "n_episodes": len(eps),
                          "n_repeats": sum(b is not None and b in bases[:i]
                                           for i, b in enumerate(bases)),
                          "only_base": bool(eps) and all(b is not None for b in bases)}
                print(f"  oracle: {len(eps)} episode(s): {oracle['instructions']} | {wall_s}s")
                out = {"trial": trial_n, "time": datetime.now().isoformat(timespec="seconds"),
                       "key": key, "category": spec["category"], "prompt": spec["prompt"],
                       "admissible": list(spec["admissible"]),
                       "spec": {k: v for k, v in spec.items() if k not in ("category", "prompt")},
                       "layout": layout, "layout_order": order,
                       "layout_slots": list(cfg.slots)}
                # Motion is attached after labelling: the baseline's report line reads a
                # first-placement split, which a per-instruction run does not have.
                labelled = label_trial(out)
                out["motion"] = task_motion(eps)
                for e in eps:
                    w = (e.get("motion") or {}).get("whole")
                    if w:
                        print(f"     motion '{e['instruction']}': "
                              f"{'OSCILLATION' if w['flag'] else 'no oscillation'} (mean "
                              f"{w['max_mean_speed']:.2f}, roll150 {w['max_roll150_speed']:.2f}, "
                              f"peak {w['max_peak_speed']:.0f})")
                note = input("Note (optional): ").strip()
                first = eps[0] if eps else None
                out.update(note=note, oracle=oracle,
                           episode=({"dir": first["dir"], **first["result"]} if first else None),
                           wall_s=wall_s,
                           config={k: getattr(cfg, k) for k in (
                               "pretrained_name_or_path", "policy_type", "actions_per_chunk",
                               "chunk_size_threshold", "episode_duration", "reps", "prior_reps",
                               "base_reps", "focus_keys", "focus_reps", "layouts", "slots", "seed",
                               "convergence_grace_period", "action_stall_timeout")})
                path = results / f"trial_{trial_n:03d}.json"
                path.write_text(json.dumps(out, indent=2, default=str))
                trials.append(out)
                print(f"  saved {path}")
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
        client.stop()
        print_stats(trials)
        print_oracle_stats(trials)
        print_comparison(trials, cfg.compare_dirs)


@draccus.wrap()
def main(cfg: OracleEvalConfig):
    main_repl(cfg)


if __name__ == "__main__":
    register_third_party_plugins()
    main()
