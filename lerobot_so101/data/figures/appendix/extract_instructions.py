"""Instruction sets of every evaluation, read from the saved results (not the code), so
the tables show what was actually run. Writes appendix_instructions.tex.
Usage: python3 extract_instructions.py <data_dir> <out_dir>"""
import collections
import glob
import json
import sys
from pathlib import Path

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
CAT_ORDER = ["prior", "base", "omission", "sequencing", "conditional", "quantity"]
OBJ_ORDER = ["banana", "toy", "pouch"]
DECOMP_FULL = D / "vlm_decompose_test/2026-09-27_03-43-57/results.json"
DECOMP = D / "vlm_decompose_test/2026-09-27_03-43-57_sample200/results.json"
FINAL = D / "vlm_final_eval_test/2026-09-29_03-03-15/results.json"
SUBTASK = D / "vlm_subtask_eval_test/2026-09-29_05-05-23/results.json"
EMBED = D / "preference_baseline/2026-10-02_00-58-18/results.json"
LIVE = {"raw VLA": "preference_eval_trials", "system": "preference_eval_system_trials",
        "oracle": "preference_eval_oracle_trials"}
MONITOR = D / "vlm_monitor_test/baseline150.json"
LIVE_MON = D / "live_monitor_trials"


def esc(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def objs(xs):
    return ", ".join(sorted(xs, key=lambda o: OBJ_ORDER.index(o) if o in OBJ_ORDER else 9))


def order_text(pairs):
    pairs = [tuple(p) for p in pairs]
    befores = collections.Counter(a for a, _ in pairs)
    afters = {b for _, b in pairs}
    # total order of all three objects -> a chain
    nodes = set(befores) | afters
    if len(nodes) == 3 and len(set(pairs)) >= 2:
        chain = sorted(nodes, key=lambda o: -sum(1 for a, _ in pairs if a == o)
                       + sum(1 for _, b in pairs if b == o))
        if all(chain.index(a) < chain.index(b) for a, b in pairs) and \
                all((chain[i], chain[i + 1]) in pairs for i in range(2)):
            return r" $\rightarrow$ ".join(chain)
    return "; ".join(f"{a} before {b}" for a, b in pairs)


def sort_key(item):
    cat = item[1].get("category")
    return CAT_ORDER.index(cat) if cat in CAT_ORDER else 99


# ---------------------------------------------------------------- offline VLM set
dec_full = json.loads(DECOMP_FULL.read_text())
dec = json.loads(DECOMP.read_text())
fin = json.loads(FINAL.read_text())
n_dec = collections.Counter(t["key"] for t in dec["trials"])
n_fin = collections.Counter(t["key"] for t in fin["trials"])
n_dec_scenes = len({t["scene"] for t in dec_full["trials"]})
n_fin_scenes = len(fin["scenes"])


def offline_target(s):
    if s.get("n"):
        return f"any {s['n']} of \\{{{objs(s['pool'])}\\}}"
    t = f"\\{{{objs(s['expected'])}\\}}"
    if s.get("order"):
        t += f"; order {order_text(s['order'])}"
    return t


rows_off = []
for key, s in sorted(dec["prompts"].items(), key=sort_key):
    rows_off.append(f"{s['category']} & {esc(s['prompt'])} & {offline_target(s)} & "
                    f"{n_dec.get(key, 0)} & {n_fin.get(key, 0)} \\\\")

# ---------------------------------------------------------------- live preference set
emb = json.loads(EMBED.read_text())
live_counts = {}
live_prompts = collections.OrderedDict()
for name, folder in LIVE.items():
    c = collections.Counter()
    for f in sorted(glob.glob(str(D / folder / "trial_*.json"))):
        t = json.loads(Path(f).read_text())
        c[t["key"]] += 1
        live_prompts.setdefault(t["key"], dict(category=t["category"], prompt=t["prompt"],
                                               admissible=t.get("admissible"),
                                               spec=t.get("spec") or {}))
    live_counts[name] = c
specs = emb["preference_prompts"]


def live_target(key, s):
    sp = specs.get(key, {}) or s.get("spec", {})
    cat = s["category"]
    if cat == "prior":
        return "unconstrained (scene prior)"
    if cat == "base":
        return key.split()[-1]
    if cat == "omission":
        return f"\\{{{objs(sp['admissible'])}\\}}"
    if cat == "sequencing":
        return f"all three; {order_text(sp['order'])}"
    if cat == "conditional":
        return f"\\{{{objs(sp['admissible'])}\\}}"
    if cat == "quantity":
        return f"{sp['n']} of \\{{{objs(sp['cls'])}\\}}"
    return ""


rows_live = []
spec_order = list(specs)
for key, s in sorted(live_prompts.items(), key=lambda it: (sort_key(it),
        spec_order.index(it[0]) if it[0] in spec_order else OBJ_ORDER.index(it[0].split()[-1]) if it[0].split()[-1] in OBJ_ORDER else 0)):
    in_emb = "\\checkmark" if (key in specs or s["category"] == "base") else ""
    rows_live.append(f"{s['category']} & {esc(s['prompt'])} & {live_target(key, s)} & {in_emb} & "
                     + " & ".join((str(live_counts[n][key]) if live_counts[n].get(key) else "--") for n in LIVE)
                     + " \\\\")
tot_live = " & ".join(str(sum(live_counts[n].values())) for n in LIVE)

# ---------------------------------------------------------------- sub-task eval
sub = json.loads(SUBTASK.read_text())["trials"]
c_sub = collections.Counter(t["instruction"] for t in sub)
rows_sub = [f"{esc(i)} & {n} \\\\" for i, n in sorted(c_sub.items(), key=lambda x: (-x[1], x[0]))]

# ---------------------------------------------------------------- monitor tests
mon = json.loads(MONITOR.read_text())["results"]
c_mon = collections.Counter((r["task_correct"], r["task_wrong"]) for r in mon)
rows_mon = [f"{esc(a)} & {esc(b)} & {n} \\\\" for (a, b), n in
            sorted(c_mon.items(), key=lambda x: (-x[1], x[0]))]
lm = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(LIVE_MON / "trial_*.json")))]
c_lm = collections.Counter((t["vla_task"], t["monitor_task"]) for t in lm)
rows_lm = [f"{esc(a)} & {esc(b)} & {'match' if a == b else 'mismatch'} & {n} \\\\"
           for (a, b), n in sorted(c_lm.items(), key=lambda x: (x[0][0] != x[0][1], x[0]))]

T = []
T.append(r"""% Generated by extract_instructions.py from the saved evaluation results.
% Requires: \usepackage{booktabs, longtable}
\chapter{Evaluation Instruction Sets}
\label{app:instructions}
All instructions refer to one scene family: a banana, a plush toy and a brown pouch on a
table with a single tray. ``Target'' is the outcome counted as correct.
""")

T.append(rf"""
\section{{Offline VLM tests: decomposition and final evaluation}}
\label{{app:instr-offline}}
The decomposition test runs each of the {len(rows_off)} instructions on {n_dec_scenes} initial
scenes ({len(dec_full['trials'])} cases); the reported results use a seeded sample of
{len(dec['trials'])} cases. The final-evaluation test crosses each instruction with
{n_fin_scenes} hand-labelled final scenes, again sampled to {len(fin['trials'])} cases. A
still image cannot show placement order, so in the final-evaluation test the sequencing
instructions are scored on the final set of objects only.
{{\small
\begin{{longtable}}{{l p{{0.42\linewidth}} p{{0.24\linewidth}} r r}}
\toprule
Category & Instruction & Target & Decomp. & Final \\
\midrule
\endhead
""" + "\n".join(rows_off) + rf"""
\midrule
 & Total & & {len(dec['trials'])} & {len(fin['trials'])} \\
\bottomrule
\end{{longtable}}}}
""")

T.append(rf"""
\section{{Preference evaluations on the robot}}
\label{{app:instr-live}}
The same preference set is used by the embedding-space baseline (raw instruction fed to
the VLA offline, {len(emb['scenes'])} scenes), the live raw-VLA baseline, the full system and the
human oracle. The live runs share one seeded queue that cycles each instruction through
three object layouts. The oracle run leaves out the prior and base controls.
{{\small
\begin{{longtable}}{{l p{{0.36\linewidth}} p{{0.22\linewidth}} c r r r}}
\toprule
Category & Instruction & Target & Emb. & VLA & System & Oracle \\
\midrule
\endhead
""" + "\n".join(rows_live) + rf"""
\midrule
 & Total trials & & & {tot_live} \\
\bottomrule
\end{{longtable}}}}
""")

T.append(rf"""
\section{{Sub-task evaluation test}}
\label{{app:instr-subtask}}
The sub-task evaluator is replayed on {len(sub)} recorded attempts. Their instructions
are the planner's own sub-tasks, including the alternative names produced by label
refinement, so this set was not written by hand.
{{\small
\begin{{longtable}}{{p{{0.6\linewidth}} r}}
\toprule
Sub-task instruction & Attempts \\
\midrule
\endhead
""" + "\n".join(rows_sub) + r"""
\bottomrule
\end{longtable}}
""")

T.append(rf"""
\section{{Execution monitor tests}}
\label{{app:instr-monitor}}
\paragraph{{Offline.}} Each of {len(mon)} clean episodes is checked twice: once with the
instruction the VLA actually executed (the monitor should continue) and once with a
deliberately wrong object (the monitor should stop).
{{\small
\begin{{longtable}}{{p{{0.4\linewidth}} p{{0.4\linewidth}} r}}
\toprule
Executed (correct) instruction & Mismatched instruction & Episodes \\
\midrule
\endhead
""" + "\n".join(rows_mon) + rf"""
\bottomrule
\end{{longtable}}}}

\paragraph{{Live.}} {len(lm)} robot trials: the VLA receives one instruction and the
monitor is told either the same one or another object.
{{\small
\begin{{longtable}}{{p{{0.32\linewidth}} p{{0.32\linewidth}} l r}}
\toprule
VLA instruction & Monitor instruction & Condition & Trials \\
\midrule
\endhead
""" + "\n".join(rows_lm) + r"""
\bottomrule
\end{longtable}}
""")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "appendix_instructions.tex").write_text("".join(T))
print("wrote", OUT / "appendix_instructions.tex")
