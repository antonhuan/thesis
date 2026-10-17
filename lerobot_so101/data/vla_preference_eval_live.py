"""
Live (physical robot) baseline eval: the RAW preference instruction is sent straight to
the VLA, with no VLM decomposition and no monitor — the robot counterpart of
vla_preference_baseline.py, which measures the same prompts offline in embedding space.

The orchestrator never hands a preference ("put everything on the tray except the
banana") to the VLA; the VLM decomposes it into the base sub-tasks the VLA was trained
on. This script measures what happens if it did. Every prompt in
vla_preference_baseline.PREFERENCE_PROMPTS (omission / sequencing / conditional /
quantity, each with its admissible set A_p) is run on the arm, interleaved with two
controls:
  * prior   "put everything on the tray" — same sentence frame, equally out of
            distribution, but no constraint and no object named. The distribution of
            its first attempt q(o) is the scene prior. A preference the VLA ignored
            scores ~ sum_{o in A_p} q(o) on reach_admissible, so the number that
            matters is the lift over that.
            (It may place one object and stop, or go on to a second.)
  * base    the single-object training instructions ("put the <obj> on the tray") —
            an execution sanity check (can it grasp each object at all), 1 rep each.
After each episode you label what the arm did:
  * attempted  the objects the arm went for, in order (a real attempt: approached and
               tried to grasp it, whether or not the grasp held), e.g. "pt"; n = none
  * placed     the objects that ended on the tray, in order (e.g. "p"; empty = none)
Out-of-distribution prompts degrade CONTROL as well as object choice (grasps fail and
the arm can oscillate even when it goes for the right object), so the preference
(INTENT) is scored on the attempted sequence and EXECUTION on what was placed:
  intent
  * reach_admissible   first attempt in A_p (the live analogue of the offline "nearest
                       base instruction is admissible")
  * reach_named        first attempt is an object the prompt NAMES (refers_to in
                       vla_preference_baseline) — keyword latching; for omission this is
                       the excluded object. n/a when the prompt names none.
  * continued          went for another object after the first
  * intent_violated    the attempted sequence breaks the preference (an excluded / non-
                       qualifying object, an order inversion, too many objects)
  * intent_complete    the attempted sequence satisfies the whole preference (all of A_p
                       for omission / conditional, n objects for quantity, every object
                       the order constrains for sequencing)
  execution
  * step_success       first object placed is in A_p
  * violated / complete  as above, on the placed sequence
  * osc / osc_first / osc_after   oscillation flag (vla_motion_metrics: 150-step rolling
                       rule OR the original velocity rule) for the whole episode, before
                       the first placement and after it — computed automatically from
                       actions.csv and stored per trial under "motion"

/stats also reports prior_exp (reach_admissible expected from the prior trials' q) and
lift = reach_admissible - prior_exp.

Spatial layout: with no usable constraint the arm tends to take the objects nearest to
it, front to back (trials 1 and 2: pouch in front, toy behind it, banana to the left ->
both went pouch, toy). So the layout is a design factor: objects go in fixed --slots,
ordered nearest-first (default front = directly ahead, back = behind it, left = off the
arm's axis), and --layouts are the objects per slot (default "ptb", "tbp", "bpt": each
object nearest once). The queue cycles each prompt through the layouts and tells you
which one to set; /layout overrides it for the next trial. Per trial:
  * nearest_first      first attempt was the nearest object (proximity explains it)
  * proximity_order    every attempt followed nearest-first order
and prior_exp becomes layout-aware: the prior's first attempt is tallied by proximity
RANK (q_rank), and a trial expects sum_{o in A_p} q_rank(rank of o in its layout).
near_adm = share of trials whose nearest object is admissible (what a pure "take the
nearest" policy would score).

Prompts run from a queue (preferences x --reps, --focus_keys x --focus_reps, prior x
--prior_reps, base x --base_reps, shuffled with --seed). /relabel N re-asks a saved
trial's labels (e.g. one skipped because it was aborted). Re-launching
with the same --results_dir resumes: (prompt, rep) pairs already on disk are skipped and
/stats covers all saved trials.

Each trial writes <results_dir>/trial_NNN.json (prompt, A_p, labels, scores, episode
result, config), and the client saves the episode in the offline format (frames +
actions.csv + recap.png) under --vla_input_dir/<prompt>/<index>_<prompt>/.

Usage (same robot / policy flags as robot_client_loop.py):
    python vla_preference_eval_live.py \\
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=arm \\
        --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 342222071104, width: 640, height: 480, fps: 30}}" \\
        --task=dummy --server_address=127.0.0.1:8080 --policy_type=pi05 \\
        --pretrained_name_or_path=ant0nh/pi05_pnp_425_25k --policy_device=cuda \\
        --actions_per_chunk=50 --chunk_size_threshold=0.7 --episode_duration=60 \\
        --reps=3
"""

import json
import logging
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import draccus

from lerobot.utils.import_utils import register_third_party_plugins
from robot_client_loop import LoopClientConfig, LoopRobotClient
from vla_motion_metrics import episode_metrics
from vla_preference_baseline import (BASE_PROMPTS, CATEGORIES, OBJECTS, PREFERENCE_PROMPTS,
                                     refers_to)

# ---------------------------------------------------------------------------
# Prompt set: the offline baseline's preferences + the prior / base controls
# ---------------------------------------------------------------------------

PRIOR_KEY = "prior everything"
PRIOR_SPECS = OrderedDict([
    (PRIOR_KEY, dict(category="prior", prompt="put everything on the tray",
                     admissible=list(OBJECTS)))])
BASE_SPECS = OrderedDict(
    (f"base {o}", dict(category="base", prompt=p, admissible=[o]))
    for o, p in BASE_PROMPTS.items())
CONTROL_CATEGORIES = ("prior", "base", "custom")


def all_prompts(cfg) -> "OrderedDict[str, dict]":
    out = OrderedDict()
    for specs in (PRIOR_SPECS, BASE_SPECS):
        out.update((k, s) for k, s in specs.items() if target_reps(k, s, cfg) > 0)
    out.update((k, s) for k, s in PREFERENCE_PROMPTS.items() if s["category"] in cfg.categories)
    return out


def target_reps(key: str, spec: dict, cfg) -> int:
    if spec["category"] == "prior":
        return cfg.prior_reps if cfg.prior_reps > 0 else 2 * cfg.reps
    if spec["category"] == "base":
        return cfg.base_reps
    if key in cfg.focus_keys:
        return max(cfg.reps, cfg.focus_reps)
    return cfg.reps


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def parse_objects(text: str) -> list[str] | None:
    """'tb' / 't,b' / 'toy banana' -> ['toy', 'banana']; '' / 'n' / 'none' -> [].
    None if a token is not an object."""
    text = text.strip().lower()
    if text in ("", "n", "none", "-"):
        return []
    tokens = text.replace(",", " ").split()
    if len(tokens) == 1 and tokens[0] not in OBJECTS:
        tokens = list(tokens[0])  # "tb" -> ["t", "b"]
    out = []
    for tok in tokens:
        match = [o for o in OBJECTS if o == tok or o[0] == tok]
        if len(match) != 1:
            return None
        out.append(match[0])
    return out


def layout_order(letters: str) -> list[str] | None:
    """'ptb' -> ['pouch', 'toy', 'banana'] (objects per slot, nearest first); None
    unless it is a permutation of OBJECTS."""
    order = parse_objects(letters)
    return order if order and sorted(order) == sorted(OBJECTS) else None


def describe_layout(order: list[str], slots: list[str]) -> str:
    return ", ".join(f"{s}={o}" for s, o in zip(slots, order))


def judge(spec: dict, seq: list[str]) -> tuple[bool, bool]:
    """(violated, complete) of an object sequence (attempted or placed) under a spec."""
    adm = set(spec["admissible"])
    cat = spec["category"]
    if cat == "omission":
        violated = any(o in spec["excluded"] for o in seq)
        complete = not violated and adm <= set(seq)
    elif cat == "sequencing":
        pos = {o: i for i, o in reversed(list(enumerate(seq)))}  # first occurrence
        # (a, b): b may only come after a has.
        violated = any(b in pos and (a not in pos or pos[a] > pos[b]) for a, b in spec["order"])
        # Complete = every object the order constrains (e.g. toy, banana for "banana
        # after toy"); unconstrained objects are not required.
        complete = not violated and {o for pair in spec["order"] for o in pair} <= set(seq)
    elif cat == "quantity":
        violated = any(o not in adm for o in seq) or len(set(seq)) > spec["n"]
        complete = not violated and len(set(seq)) == spec["n"]
    else:  # conditional, base, prior, custom
        violated = any(o not in adm for o in seq)
        complete = not violated and adm <= set(seq)
    return violated, complete


def score(spec: dict, attempted: list[str], placed: list[str],
          order: list[str] | None = None) -> dict:
    """Score one trial's labels against a preference spec (see module docstring).
    `order`: the layout's objects nearest-first (None if unknown)."""
    adm = set(spec["admissible"])
    reach = attempted[0] if attempted else None
    named = refers_to(spec) if spec["category"] not in ("prior", "quantity") else []
    if set(named) == set(OBJECTS):
        named = []
    intent_violated, intent_complete = judge(spec, attempted)
    violated, complete = judge(spec, placed)
    return {"reach_admissible": reach in adm if reach else False,
            "reach_named": (reach in named) if named else None,
            "continued": len(attempted) > 1,
            "intent_violated": intent_violated, "intent_complete": intent_complete,
            "step_success": bool(placed) and placed[0] in adm,
            "violated": violated, "complete": complete,
            "nearest_first": (reach == order[0]) if order else None,
            "proximity_order": (bool(attempted) and attempted == order[:len(attempted)])
                               if order else None,
            "near_adm": (order[0] in adm) if order else None}


def trial_spec(t: dict) -> dict:
    return {**t.get("spec", {}), "category": t["category"], "prompt": t["prompt"],
            "admissible": t["admissible"]}


def normalize_trial(t: dict) -> dict:
    """Bring a loaded trial up to the current format: 'attempted' (older trials had only
    'reach'), scores recomputed with the current scoring, motion metrics computed from
    the episode's actions.csv if missing (in memory only)."""
    if t.get("label_ok"):
        if "attempted" not in t:
            t["attempted"] = [t["reach"]] if t.get("reach") else []
        t["scores"] = score(trial_spec(t), t["attempted"], t.get("placed") or [],
                            t.get("layout_order"))
    if t.get("motion") is None:
        t["motion"] = motion_for(t.get("episode", {}).get("dir"))
    return t


def motion_for(episode_dir: str | None) -> dict | None:
    path = Path(episode_dir) / "actions.csv" if episode_dir else None
    if path is None or not path.exists():
        return None
    try:
        return episode_metrics(path)
    except Exception as e:  # noqa: BLE001 - metrics must never break a trial
        logging.warning(f"motion metrics failed for {path}: {e}")
        return None


def prior_distribution(trials: list[dict]) -> dict[str, float] | None:
    """q(o): share of prior ('put everything') trials whose first reach was o (a 'none'
    reach counts in the denominator). None until a prior trial is labelled."""
    prior = [t for t in trials if t["category"] == "prior"]
    if not prior:
        return None
    return {o: sum(t.get("reach") == o for t in prior) / len(prior) for o in OBJECTS}


def prior_rank_distribution(trials: list[dict]) -> list[float] | None:
    """q_rank[r]: share of prior trials (with a known layout) whose first attempt was the
    rank-r object (0 = nearest); a 'none' attempt counts in the denominator."""
    prior = [t for t in trials if t["category"] == "prior" and t.get("layout_order")]
    if not prior:
        return None
    return [sum(bool(t.get("attempted")) and t["attempted"][0] == t["layout_order"][r]
                for t in prior) / len(prior) for r in range(len(OBJECTS))]


def expected_from_prior(t: dict, q: dict | None, q_rank: list[float] | None) -> float | None:
    """reach_admissible this trial would score if the VLA ignored the prompt: from the
    rank prior when the layout is known, else the object prior."""
    if q_rank is not None and t.get("layout_order"):
        return sum(q_rank[t["layout_order"].index(o)] for o in t["admissible"])
    if q is not None:
        return sum(q[o] for o in t["admissible"] if o in q)
    return None


def summarize(trials: list[dict]) -> list[tuple[str, int, dict]]:
    """[(group, n, {metric: rate | None})] per category, then per prompt. prior_exp is
    the reach_admissible a prompt would score if the VLA ignored it (sum of q over A_p);
    lift = reach_admissible - prior_exp. reach_named averages only over trials whose
    prompt names an object."""
    q = prior_distribution(trials)
    q_rank = prior_rank_distribution(trials)
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for cat in ("prior", "base", *CATEGORIES, "custom"):
        groups[cat] = [t for t in trials if t["category"] == cat]
    groups["ALL preferences"] = [t for t in trials if t["category"] not in CONTROL_CATEGORIES]
    for t in trials:
        groups.setdefault(f"  {t['key']}", []).append(t)

    def mean(xs):
        xs = [float(x) for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def osc(t, phase):
        mo = t.get("motion") or {}
        ph = mo.get(phase)
        return None if ph is None else ph["flag" if phase == "whole" else "flag_rolling"]

    rows = []
    for g, ts in groups.items():
        if not ts:
            continue
        m = {k: mean(t["scores"].get(k) for t in ts) for k in (
            "reach_admissible", "reach_named", "continued", "intent_violated",
            "intent_complete", "step_success", "violated", "complete", "nearest_first",
            "proximity_order", "near_adm")}
        for phase in ("whole", "first", "after"):
            m[f"osc_{phase}"] = mean(osc(t, phase) for t in ts)
        m["n_after"] = sum(osc(t, "after") is not None for t in ts)
        m["prior_exp"] = (mean(expected_from_prior(t, q, q_rank) for t in ts)
                          if any(t["category"] != "prior" for t in ts) else None)
        m["lift"] = (m["reach_admissible"] - m["prior_exp"]
                     if m["prior_exp"] is not None else None)
        rows.append((g, len(ts), m))
    return rows


# ---------------------------------------------------------------------------
# Client: episode directories named by prompt key + end-of-episode release
# ---------------------------------------------------------------------------


@dataclass
class PreferenceEvalConfig(LoopClientConfig):
    episode_duration: float = 60
    reps: int = 3                      # runs of each prompt in the queue
    shuffle: bool = True               # shuffle the queue (seeded) so prompts interleave
    seed: int = 0
    # Controls (see module docstring). prior_reps=0 means 2 x reps: every preference is
    # compared against the prior's first-reach distribution, so it gets more runs.
    prior_reps: int = 0
    base_reps: int = 1                 # single-object execution check; 0 disables
    # Prompts that decide a question get more runs: "one" vs "two" objects (with the
    # prior) separates responding to the number from just continuing to pick.
    focus_keys: list[str] = field(default_factory=lambda: ["qty one object", "qty two objects"])
    focus_reps: int = 5
    categories: list[str] = field(default_factory=lambda: list(CATEGORIES))
    # Spatial layout (see module docstring): slots nearest-first, and the objects per
    # slot for each layout (letters in slot order). Each object is nearest once.
    slots: list[str] = field(default_factory=lambda: ["front", "back", "left"])
    layouts: list[str] = field(default_factory=lambda: ["ptb", "tbp", "bpt"])
    # End-of-episode release (same sequence as vlm_monitor_live.py): open the gripper to
    # gripper_open_pos over release_steps, dwell, raise shoulder_lift up to
    # release_lift_deg toward home, then go home. release_lift_deg=0 skips the lift.
    gripper_open_pos: float = 40.0
    release_steps: int = 15
    release_dwell_s: float = 0.5
    release_lift_deg: float = 20.0
    release_lift_steps: int = 15
    results_dir: str = "preference_eval_trials"
    vla_input_dir: str = "preference_eval_trials/episodes"


class PreferenceEvalClient(LoopRobotClient):
    """LoopRobotClient whose episode directories are named after the prompt key
    (the base class names them by the word after "the", which for "put everything on
    the tray except ..." would be "tray" for every omission prompt)."""

    def __init__(self, config):
        self.prompt_tag: str | None = None
        super().__init__(config)

    def _extract_object(self, task):
        return self._slugify_task(self.prompt_tag) if self.prompt_tag else super()._extract_object(task)

    def _gripper_key(self):
        return next((k for k in self.robot.action_features if k.split(".")[0] == "gripper"), None)

    def release_gripper(self):
        """Smoothly open the gripper in place (other joints held), then dwell."""
        cfg = self.config
        g = self._gripper_key()
        if g is None:
            self.logger.warning("No gripper joint found; skipping release.")
            return
        current = self._capture_current_position()
        start, end = current.get(g, 0.0), float(cfg.gripper_open_pos)
        for step in range(1, cfg.release_steps + 1):
            target = dict(current)
            target[g] = start + step / cfg.release_steps * (end - start)
            self.robot.send_action(target)
            time.sleep(cfg.home_step_dt)
        time.sleep(cfg.release_dwell_s)

    def lift_away(self):
        """Raise shoulder_lift up to release_lift_deg toward home (gripper stays open)."""
        cfg = self.config
        if cfg.release_lift_deg <= 0:
            return
        lift = next((k for k in self.robot.action_features
                     if k.split(".")[0] == "shoulder_lift"), None)
        if lift is None or lift not in self.home_position:
            return
        current = self._capture_current_position()
        start, home = current.get(lift, 0.0), self.home_position[lift]
        delta = max(-cfg.release_lift_deg, min(cfg.release_lift_deg, home - start))
        for step in range(1, cfg.release_lift_steps + 1):
            target = dict(current)
            target[lift] = start + step / cfg.release_lift_steps * delta
            self.robot.send_action(target)
            time.sleep(cfg.home_step_dt)

    def release_and_home(self):
        self.execution_gate.set()
        self.release_gripper()
        self.lift_away()
        self.go_home()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

HELP = """Commands:
  /run  (or Enter)   run the current prompt (Enter during the episode aborts it)
  /list              all prompts with their admissible set and runs done
  /pref <key|#>      run this prompt next (key as in /list, e.g. 'omit banana', or its #)
  /task <text>       a custom prompt (set its admissible objects with /admissible)
  /admissible b,t    admissible objects for the custom prompt
  /layout <letters>  the layout for the next trial, objects per slot nearest-first
                     (e.g. /layout tbp); /layout alone shows the slots
  /skip              skip the current queue item
  /queue             show the remaining queue
  /relabel <trial#>  re-enter a saved trial's labels (e.g. an aborted / skipped one)
  /duration <s>      /home     /release (open gripper, lift away, then home)
  /stats     /config     /help     /quit
  (every episode ends with: open gripper -> lift away -> home; reset the scene after)"""


def load_trials(results: Path) -> list[dict]:
    trials = []
    for p in sorted(results.glob("trial_*.json")):
        try:
            trials.append(normalize_trial(json.loads(p.read_text())))
        except (OSError, json.JSONDecodeError) as e:
            logging.warning(f"Skipping unreadable {p}: {e}")
    return trials


def done_counts(trials: list[dict]) -> dict[str, int]:
    """Labelled runs per prompt key."""
    done: dict[str, int] = {}
    for t in trials:
        if t.get("label_ok"):
            done[t["key"]] = done.get(t["key"], 0) + 1
    return done


def build_queue(prompts, cfg: PreferenceEvalConfig,
                trials: list[dict]) -> list[tuple[str, str]]:
    """(key, layout) items: every prompt x its target reps, rep r of the i-th prompt in
    layout (i + r) mod n (so each prompt cycles the layouts and they are not all started
    on the same one), minus labelled runs already saved (matched by layout when known)."""
    out = []
    for i, (k, s) in enumerate(prompts.items()):
        want = [cfg.layouts[(i + r) % len(cfg.layouts)]
                for r in range(target_reps(k, s, cfg))]
        for t in trials:
            if t.get("label_ok") and t["key"] == k and want:
                want.remove(t["layout"] if t.get("layout") in want else want[-1])
        out += [(k, lay) for lay in want]
    items = out
    if cfg.shuffle:
        random.Random(cfg.seed).shuffle(items)
    return items


def _dedupe(seq: list[str]) -> list[str]:
    """Collapse consecutive repeats (several grasp tries at one object = one attempt)."""
    return [o for i, o in enumerate(seq) if i == 0 or seq[i - 1] != o]


def ask_labels() -> tuple[list[str], list[str]] | None:
    """Prompt for attempted + placed labels. None if the trial is skipped."""
    letters = ", ".join(f"{o[0]}={o}" for o in OBJECTS)
    while True:
        ans = input(f"Objects the arm went for, in order [e.g. 'pt' ({letters}); "
                    "n = none; s = skip trial]: ")
        if ans.strip().lower() == "s":
            return None
        attempted = parse_objects(ans)
        if attempted is not None:
            attempted = _dedupe(attempted)
            break
        print(f"  objects as letters ({letters}), n, or s")
    while True:
        placed = parse_objects(input("Objects on the tray at the end, in order "
                                     "[e.g. 'p'; Enter = none]: "))
        if placed is None:
            print(f"  objects as letters ({letters})")
        elif not set(placed) <= set(attempted):
            print(f"  placed {placed} but attempted only {attempted or 'none'} — "
                  "re-enter (or 's' at the first question to skip)")
            return ask_labels()
        else:
            return attempted, placed


def report_trial(sc: dict, attempted: list[str], placed: list[str], motion: dict | None):
    named = ", NAMED object first" if sc["reach_named"] else ""
    if sc.get("nearest_first"):
        named += ", NEAREST object first" + (" (proximity order)" if sc["proximity_order"]
                                              and len(attempted) > 1 else "")
    print(f"  -> intent: went for {attempted or 'none'} "
          f"({'admissible' if sc['reach_admissible'] else 'NOT admissible'} first{named}"
          f"{', continued' if sc['continued'] else ''}"
          f"{', VIOLATED' if sc['intent_violated'] else ''}"
          f"{', complete' if sc['intent_complete'] else ''})")
    print(f"     execution: placed {placed or 'none'} (step {'OK' if sc['step_success'] else 'fail'}"
          f"{', VIOLATED' if sc['violated'] else ''}{', complete' if sc['complete'] else ''})")
    if motion:
        w, a = motion["whole"], motion["after"]
        print(f"     motion: {'OSCILLATION' if w['flag'] else 'no oscillation'} (mean "
              f"{w['max_mean_speed']:.2f}, roll150 {w['max_roll150_speed']:.2f}, peak "
              f"{w['max_peak_speed']:.0f})"
              + (f"; first placement at t={motion['split_t']}s, after it roll150 "
                 f"{a['max_roll150_speed']:.2f}{' OSC' if a['flag_rolling'] else ''}"
                 if a else "; no placement detected"))


def ask_layout(current: str | None, slots: list[str]) -> str | None:
    """Ask the layout (letters per slot, nearest-first); Enter keeps `current`."""
    while True:
        ans = input(f"Layout, objects in slots {'/'.join(slots)} nearest-first "
                    f"[e.g. 'ptb'; Enter = {current or 'unknown'}]: ").strip().lower()
        if not ans:
            return current
        if layout_order(ans):
            return ans
        print("  three letters, one per object, e.g. 'ptb'")


def label_trial(out: dict) -> bool:
    """Ask the labels for a trial record and fill in attempted / placed / scores."""
    labels = ask_labels()
    out["label_ok"] = labels is not None
    if labels is None:
        return False
    attempted, placed = labels
    sc = score(trial_spec(out), attempted, placed, out.get("layout_order"))
    out.update(reach=attempted[0] if attempted else None, attempted=attempted,
               placed=placed, scores=sc)
    report_trial(sc, attempted, placed, out.get("motion"))
    return True


def print_stats(trials: list[dict]):
    rows = summarize([t for t in trials if t.get("label_ok")])
    if not rows:
        print("  no labelled trials yet")
        return
    labelled = [t for t in trials if t.get("label_ok")]
    q = prior_distribution(labelled)
    q_rank = prior_rank_distribution(labelled)
    if q is None:
        print("  (no prior trials yet — prior_exp / lift need 'put everything' runs)")
    else:
        print("  prior first attempt by object q: "
              + ", ".join(f"{o} {p:.0%}" for o, p in q.items()))
    if q_rank is not None:
        print("  prior first attempt by proximity rank q_rank: "
              + ", ".join(f"#{r + 1} {p:.0%}" for r, p in enumerate(q_rank))
              + "   (prior_exp uses this when the trial's layout is known)")

    def pct(x, w, signed=False):
        return f"{'-':>{w}}" if x is None else f"{x:+{w}.0%}" if signed else f"{x:{w}.0%}"

    print("\n  INTENT (attempted objects)")
    print(f"  {'group':34} {'n':>3}  {'reach adm':>9} {'prior exp':>9} {'lift':>6} "
          f"{'near adm':>8} {'near 1st':>8} {'named':>6} {'contin.':>7} {'violated':>8} "
          f"{'complete':>8}")
    for g, n, m in rows:
        print(f"  {g:34} {n:3d}  {pct(m['reach_admissible'], 9)} {pct(m['prior_exp'], 9)} "
              f"{pct(m['lift'], 6, True)} {pct(m['near_adm'], 8)} {pct(m['nearest_first'], 8)} "
              f"{pct(m['reach_named'], 6)} {pct(m['continued'], 7)} "
              f"{pct(m['intent_violated'], 8)} {pct(m['intent_complete'], 8)}")
    print("\n  EXECUTION (placed objects, oscillation: whole / before / after 1st placement)")
    print(f"  {'group':34} {'n':>3}  {'step ok':>8} {'violated':>8} {'complete':>8} "
          f"{'osc':>6} {'osc 1st':>7} {'osc aft':>7} {'n aft':>5}")
    for g, n, m in rows:
        print(f"  {g:34} {n:3d}  {pct(m['step_success'], 8)} {pct(m['violated'], 8)} "
              f"{pct(m['complete'], 8)} {pct(m['osc_whole'], 6)} {pct(m['osc_first'], 7)} "
              f"{pct(m['osc_after'], 7)} {m['n_after']:5d}")
    by_layout: "OrderedDict[str, list[dict]]" = OrderedDict()
    for t in labelled:
        by_layout.setdefault(t.get("layout") or "?", []).append(t)
    print("\n  LAYOUT (nearest 1st / proximity order / first attempt admissible)")
    print(f"  {'layout':8} {'prior n':>7} {'near 1st':>8} {'prox ord':>8}   "
          f"{'pref n':>6} {'near 1st':>8} {'prox ord':>8} {'reach adm':>9}")

    def rate(ts, k):
        xs = [t["scores"].get(k) for t in ts if t["scores"].get(k) is not None]
        return sum(xs) / len(xs) if xs else None

    for lay, ts in by_layout.items():
        pr = [t for t in ts if t["category"] == "prior"]
        pf = [t for t in ts if t["category"] not in CONTROL_CATEGORIES]
        print(f"  {lay:8} {len(pr):7d} {pct(rate(pr, 'nearest_first'), 8)} "
              f"{pct(rate(pr, 'proximity_order'), 8)}   {len(pf):6d} "
              f"{pct(rate(pf, 'nearest_first'), 8)} {pct(rate(pf, 'proximity_order'), 8)} "
              f"{pct(rate(pf, 'reach_admissible'), 9)}")


def main_repl(cfg: PreferenceEvalConfig):
    logging.basicConfig(level=logging.INFO)
    results = Path(cfg.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    prompts = all_prompts(cfg)
    keys = list(prompts)
    trials = load_trials(results)
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

    client = PreferenceEvalClient(cfg)
    if not client.connect_server():
        client.logger.error("Could not connect to policy server. Exiting.")
        client.stop()
        return

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
              f"{len(trials)}  duration={cfg.episode_duration}s  "
              f"policy={cfg.pretrained_name_or_path}")

    print("=" * 64 + "\nVLA PREFERENCE BASELINE (raw preference -> VLA, no VLM)\n"
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
                      f"| A_p={spec['admissible']} | layout {layout} (Enter aborts) ---")
                client.prompt_tag = key
                t0 = time.time()
                result = client.run_episode(spec["prompt"], enable_abort_listener=True)
                episode_dir = str(client._vla_input_dir)
                client.release_and_home()
                out = {"trial": trial_n, "time": datetime.now().isoformat(timespec="seconds"),
                       "key": key, "category": spec["category"], "prompt": spec["prompt"],
                       "admissible": list(spec["admissible"]),
                       "spec": {k: v for k, v in spec.items() if k not in ("category", "prompt")},
                       "layout": layout, "layout_order": order,
                       "layout_slots": list(cfg.slots),
                       "motion": motion_for(episode_dir)}
                labelled = label_trial(out)
                note = input("Note (optional): ").strip()
                out.update(note=note, episode={"dir": episode_dir, **asdict(result)},
                           wall_s=round(time.time() - t0, 1),
                           config={k: getattr(cfg, k) for k in (
                               "pretrained_name_or_path", "policy_type", "actions_per_chunk",
                               "chunk_size_threshold", "episode_duration", "reps", "prior_reps",
                               "base_reps", "focus_keys", "focus_reps", "layouts", "slots", "seed",
                               "convergence_grace_period", "action_stall_timeout")})
                path = results / f"trial_{trial_n:03d}.json"
                path.write_text(json.dumps(out, indent=2, default=str))
                trials.append(out)
                print(f"  saved {path} (episode frames: {episode_dir})")
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


@draccus.wrap()
def main(cfg: PreferenceEvalConfig):
    main_repl(cfg)


if __name__ == "__main__":
    register_third_party_plugins()
    main()
