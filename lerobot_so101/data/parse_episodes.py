#!/usr/bin/env python3
"""Parse low-level VLA episodes out of the ``runs/`` action logs.

Every SmolVLA episode is logged to ``runs/<run>/<NN>/actions.log`` by
``robot_client_loop.py``. Each episode is the execution of one decomposed
sub-task -- the exact prompt string the VLA received -- followed by the
per-timestep 6-joint targets it produced. This script splits each log into
individual episodes, extracts the action trajectories, correlates each episode
to its high-level context in the sibling ``task.log`` (high-level prompt, round,
and the VLM's own success judgement), and writes:

  * episode_analysis/manifest.csv       one row per sub-task episode, keyed to its
                                        high-level episode via ``hl_episode_id``,
                                        with a blank ``manual_success`` to hand-label.
  * episode_analysis/episodes.csv       one row per high-level episode (each
                                        ``runs/<run>/<NN>/``): the high-level prompt,
                                        round/sub-task counts, and the orchestrator's
                                        terminal verdict (complete / incomplete /
                                        aborted / unknown), plus a blank
                                        ``manual_success`` to hand-label.
  * episode_analysis/actions/<id>.csv   the full T x 6 action trajectory.

Each ``runs/<run>/<NN>/`` is one high-level episode: a single high-level prompt the
VLM orchestrator decomposes -- possibly re-decomposing over several rounds -- into
sub-tasks, each of which is one SmolVLA episode in ``actions.log``.

Run it from ``lerobot_so101/data/``:

    python parse_episodes.py

Only the Aug-19/Aug-20 runs carry an ``actions.log``; earlier runs predate
action logging and are skipped (reported in the summary).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Joint order as emitted by robot_client_loop._format_action.
JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

RUNS_DIR = Path("runs")
OUT_DIR = Path("episode_analysis")

# --- line patterns -----------------------------------------------------------
RE_EP_START = re.compile(r"=== episode start \| task: '(?P<task>.*)' \| duration: (?P<dur>[\d.]+)s ===")
RE_EP_END = re.compile(
    r"=== episode end \| (?P<dur>[\d.]+)s \| converged=(?P<conv>\w+) \| "
    r"actions=(?P<n>\d+)(?: \| reason=(?P<reason>\w+))? ==="
)
RE_ACTION = re.compile(
    r"\[action #(?P<n>\d+)\] t=(?P<t>[\d.]+)s step=(?P<step>\d+) "
    r"queue=(?P<queue>\d+) dmax=(?P<dmax>n/a|[\d.]+) \| (?P<joints>.+)$"
)

# task.log patterns
# The round suffix "(round N)" is absent in the oldest runs -- keep it optional.
RE_DECOMPOSE = re.compile(
    r"Decomposing high-level prompt(?: \(round (?P<round>\d+)\))?: '(?P<prompt>.*)'"
)
RE_EXECUTING = re.compile(r"\[\d+\] Executing sub-task: '(?P<task>.*?)' \(")
RE_JUDGEMENT = re.compile(r"VLM judgement: success=(?P<success>\w+) \| (?P<reason>.*)$")

# Terminal high-level-episode outcome patterns. Every NN dir ends in exactly one of:
#   complete (old + new phrasings), incomplete (retries exhausted), or operator abort.
# Guard with LOG_PREFIX so the multi-line retry-context prose the planner is fed
# (which also contains "judged INCOMPLETE") is not mistaken for a real log line.
RE_HL_COMPLETE_OLD = re.compile(r"High-level task complete: '(?P<prompt>.*)'\s*$")
RE_HL_COMPLETE_NEW = re.compile(
    r"High-level task '(?P<prompt>.*)' judged COMPLETE after (?P<rounds>\d+) round"
)
RE_HL_INCOMPLETE = re.compile(
    r"High-level task '(?P<prompt>.*)' judged INCOMPLETE after (?P<rounds>\d+) "
    r"round\(s\) and no retry rounds remain: (?P<reason>.*)$"
)
RE_HL_ABORT = re.compile(r"abandoning high-level task '(?P<prompt>.*?)'")
# Not anchored to line-start: an interactive input prompt (e.g. "[REPLAN] ...: ")
# can be flushed onto the same line just before the real log record.
LOG_PREFIX = re.compile(r"(?:INFO|WARNING|DEBUG|ERROR) \d{4}-\d\d-\d\d \d\d:\d\d:\d\d [\w.]+:\d+ ")


def parse_joints(segment: str) -> dict[str, float]:
    """Turn 'shoulder_pan.pos=-4.97 gripper.pos=+1.12 ...' into a name->float dict."""
    out: dict[str, float] = {}
    for tok in segment.strip().split():
        name, _, val = tok.partition("=")
        name = name.replace(".pos", "")
        out[name] = float(val)
    return out


@dataclass
class Episode:
    index: int  # 0-based order within the actions.log file
    task: str
    duration_s: str = ""
    converged: str = ""
    termination_reason: str = ""  # timeout | convergence | stall | aborted | external
    num_actions: int = 0
    truncated: bool = False
    actions: list[dict] = field(default_factory=list)  # rows: action_n,t,step,queue,dmax,+joints
    # filled in from task.log
    high_level_prompt: str = ""
    round: str = ""
    vlm_success: str = ""
    vlm_reason: str = ""


def parse_actions_log(path: Path) -> list[Episode]:
    """Split one actions.log into ordered Episodes with their action rows."""
    episodes: list[Episode] = []
    cur: Episode | None = None

    for line in path.read_text().splitlines():
        m = RE_EP_START.search(line)
        if m:
            cur = Episode(index=len(episodes), task=m.group("task"), duration_s=m.group("dur"))
            episodes.append(cur)
            continue

        m = RE_EP_END.search(line)
        if m and cur is not None:
            cur.duration_s = m.group("dur")
            cur.converged = m.group("conv")
            cur.termination_reason = m.group("reason") or ""
            cur.num_actions = int(m.group("n"))
            cur = None
            continue

        m = RE_ACTION.search(line)
        if m and cur is not None:
            joints = parse_joints(m.group("joints"))
            row = {
                "action_n": int(m.group("n")),
                "t": m.group("t"),
                "step": int(m.group("step")),
                "queue": int(m.group("queue")),
                "dmax": "" if m.group("dmax") == "n/a" else m.group("dmax"),
            }
            row.update({j: joints.get(j, "") for j in JOINTS})
            cur.actions.append(row)

    # A final episode with no "episode end" marker is truncated: fill from rows.
    for ep in episodes:
        if ep.converged == "" and not ep.num_actions:
            ep.truncated = True
            ep.num_actions = len(ep.actions)
    return episodes


def parse_task_log(path: Path) -> list[dict]:
    """Ordered sub-task executions from task.log, each with HL prompt, round, judgement."""
    executions: list[dict] = []
    cur_prompt = ""
    cur_round = ""
    pending: dict | None = None  # execution awaiting its VLM judgement

    for line in path.read_text().splitlines():
        m = RE_DECOMPOSE.search(line)
        if m:
            cur_prompt = m.group("prompt")
            cur_round = m.group("round") or "1"  # old runs omit the round suffix
            continue

        m = RE_EXECUTING.search(line)
        if m:
            pending = {
                "task": m.group("task"),
                "high_level_prompt": cur_prompt,
                "round": cur_round,
                "vlm_success": "",
                "vlm_reason": "",
            }
            executions.append(pending)
            continue

        m = RE_JUDGEMENT.search(line)
        if m and pending is not None:
            pending["vlm_success"] = m.group("success")
            pending["vlm_reason"] = m.group("reason")
            pending = None
            continue

    return executions


def parse_high_level(path: Path) -> dict:
    """Roll a task.log up to one high-level-episode record.

    Returns the high-level prompt, the number of decompose rounds, and the
    orchestrator's terminal verdict for the whole high-level task:
      * ``complete``   -- judged done (old or new phrasing)      -> hl_success=True
      * ``incomplete`` -- retries exhausted, still not done      -> hl_success=False
      * ``aborted``    -- operator interjected an ABORT          -> hl_success=False
      * ``unknown``    -- log cut off before any verdict         -> hl_success=""
    The last terminal line wins (there is only ever one per run in practice).
    """
    high_level_prompt = ""
    max_round = 0
    outcome = "unknown"
    hl_success = ""
    hl_reason = ""
    terminal_rounds = ""

    for line in path.read_text().splitlines():
        pref = LOG_PREFIX.search(line)
        if not pref:
            continue  # skip planner prose / VLM output blocks that lack a log prefix
        body = line[pref.end():]

        m = RE_DECOMPOSE.search(body)
        if m:
            high_level_prompt = m.group("prompt")
            max_round = max(max_round, int(m.group("round") or "1"))
            continue

        m = RE_HL_COMPLETE_OLD.search(body)
        if m:
            high_level_prompt = high_level_prompt or m.group("prompt")
            outcome, hl_success = "complete", "True"
            continue

        m = RE_HL_COMPLETE_NEW.search(body)
        if m:
            high_level_prompt = high_level_prompt or m.group("prompt")
            outcome, hl_success, terminal_rounds = "complete", "True", m.group("rounds")
            continue

        m = RE_HL_INCOMPLETE.search(body)
        if m:
            high_level_prompt = high_level_prompt or m.group("prompt")
            outcome, hl_success = "incomplete", "False"
            hl_reason, terminal_rounds = m.group("reason"), m.group("rounds")
            continue

        m = RE_HL_ABORT.search(body)
        if m:
            high_level_prompt = high_level_prompt or m.group("prompt")
            outcome, hl_success, hl_reason = "aborted", "False", "operator abort"
            continue

    num_rounds = int(terminal_rounds) if terminal_rounds else max_round
    return {
        "high_level_prompt": high_level_prompt,
        "num_rounds": num_rounds,
        "hl_outcome": outcome,
        "hl_success": hl_success,
        "hl_reason": hl_reason,
    }


def correlate(episodes: list[Episode], executions: list[dict]) -> list[str]:
    """Attach task.log context to episodes by order; return any mismatch warnings."""
    warnings: list[str] = []
    for ep in episodes:
        if ep.index >= len(executions):
            warnings.append(f"episode {ep.index} ('{ep.task}') has no matching task.log sub-task")
            continue
        ex = executions[ep.index]
        if ex["task"] != ep.task:
            warnings.append(
                f"prompt mismatch at episode {ep.index}: "
                f"actions.log='{ep.task}' vs task.log='{ex['task']}' (paired by order anyway)"
            )
        ep.high_level_prompt = ex["high_level_prompt"]
        ep.round = ex["round"]
        ep.vlm_success = ex["vlm_success"]
        ep.vlm_reason = ex["vlm_reason"]
    return warnings


def main() -> None:
    actions_logs = sorted(RUNS_DIR.glob("*/*/actions.log"))
    run_dirs_total = sorted({p.parent.parent for p in RUNS_DIR.glob("*/*/") if p.parent.parent})
    dirs_with_log = {p.parent for p in actions_logs}

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "actions").mkdir(exist_ok=True)

    manifest_rows: list[dict] = []
    episode_rows: list[dict] = []  # one row per high-level episode (NN dir)
    all_warnings: list[str] = []
    skipped_no_log = 0
    truncated_ids: list[str] = []

    # Count run-subdirs that have no actions.log (older runs).
    for run in sorted(RUNS_DIR.iterdir()):
        if not run.is_dir():
            continue
        for sub in sorted(run.iterdir()):
            if sub.is_dir() and sub not in dirs_with_log:
                skipped_no_log += 1

    for log_path in actions_logs:
        subdir = log_path.parent          # runs/<run>/<NN>
        run = subdir.parent.name
        nn = subdir.name

        hl_episode_id = f"{run}__{nn}"

        episodes = parse_actions_log(log_path)
        task_log = subdir / "task.log"
        high_level: dict = {}
        if task_log.exists():
            executions = parse_task_log(task_log)
            all_warnings += [f"{run}/{nn}: {w}" for w in correlate(episodes, executions)]
            high_level = parse_high_level(task_log)

        for ep in episodes:
            episode_id = f"{run}__{nn}__ep{ep.index}"
            actions_rel = Path("actions") / f"{episode_id}.csv"

            # per-episode trajectory
            with (OUT_DIR / actions_rel).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["action_n", "t", "step", "queue", "dmax", *JOINTS])
                w.writeheader()
                w.writerows(ep.actions)

            if ep.truncated:
                truncated_ids.append(episode_id)

            manifest_rows.append(
                {
                    "episode_id": episode_id,
                    "hl_episode_id": hl_episode_id,
                    "run": run,
                    "subdir": nn,
                    "episode_index": ep.index,
                    "vla_prompt": ep.task,
                    "high_level_prompt": ep.high_level_prompt,
                    "round": ep.round,
                    "num_actions": ep.num_actions,
                    "duration_s": ep.duration_s,
                    "converged": ep.converged,
                    "termination_reason": ep.termination_reason,
                    "vlm_success": ep.vlm_success,
                    "vlm_reason": ep.vlm_reason,
                    "actions_csv": str(actions_rel),
                    "manual_success": "",  # <- hand-label this column
                }
            )

        # High-level rollup: one row per NN, summarising its sub-tasks + final verdict.
        total_duration = sum(float(e.duration_s) for e in episodes if e.duration_s)
        episode_rows.append(
            {
                "hl_episode_id": hl_episode_id,
                "run": run,
                "subdir": nn,
                "high_level_prompt": high_level.get("high_level_prompt", ""),
                "num_rounds": high_level.get("num_rounds", ""),
                "num_subtasks": len(episodes),
                "num_subtasks_success": sum(1 for e in episodes if e.vlm_success == "True"),
                "total_actions": sum(e.num_actions for e in episodes),
                "total_duration_s": f"{total_duration:.1f}",
                "hl_outcome": high_level.get("hl_outcome", "unknown"),
                "hl_success": high_level.get("hl_success", ""),
                "hl_reason": high_level.get("hl_reason", ""),
                "manual_success": "",  # <- hand-label this column
            }
        )

    manifest_path = OUT_DIR / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    episodes_path = OUT_DIR / "episodes.csv"
    with episodes_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        w.writeheader()
        w.writerows(episode_rows)

    # --- summary -------------------------------------------------------------
    outcomes = Counter(r["hl_outcome"] for r in episode_rows)
    print(f"Parsed {len(manifest_rows)} sub-task episodes from {len(actions_logs)} actions.log files.")
    print(
        f"Rolled up {len(episode_rows)} high-level episode(s) "
        f"(complete/incomplete/aborted/unknown = "
        f"{outcomes['complete']}/{outcomes['incomplete']}/{outcomes['aborted']}/{outcomes['unknown']})."
    )
    print(f"Skipped {skipped_no_log} run-subdir(s) with no actions.log (pre-logging runs).")
    print(f"Wrote {manifest_path}, {episodes_path}, and {len(manifest_rows)} "
          f"per-episode CSVs to {OUT_DIR/'actions'}/.")
    if truncated_ids:
        print(f"{len(truncated_ids)} truncated episode(s) (no 'episode end' marker): "
              + ", ".join(truncated_ids))
    if all_warnings:
        print(f"\n{len(all_warnings)} correlation warning(s):")
        for w in all_warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
