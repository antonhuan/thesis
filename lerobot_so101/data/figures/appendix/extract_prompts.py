"""Pull the full-system prompts out of the orchestrator sources (via ast, no imports)
and write a LaTeX appendix + a Markdown copy. Usage: extract_prompts.py <data_dir> <out_dir>"""
import ast
import sys
from pathlib import Path

data, out = Path(sys.argv[1]), Path(sys.argv[2])
SRC = {m: (data / f"{m}.py").read_text() for m in
       ("vlm", "vlm_robot_orchestrator", "vlm_monitor_tune")}
TREES = {m: ast.parse(s) for m, s in SRC.items()}


def const(mod, name):
    for node in TREES[mod].body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{mod}.{name}")


def func(mod, name, extra_ns=None):
    for node in TREES[mod].body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"Image": type("Image", (), {"Image": object}), **(extra_ns or {})}
            code = ast.Module(body=[node], type_ignores=[])
            exec(compile(code, f"{mod}.py", "exec"), ns)
            return ns[name]
    raise KeyError(f"{mod}.{name}")


def render(content):
    """A user-turn content list -> text, images shown as <image>."""
    parts = []
    for c in content:
        parts.append("<image>" if c["type"] == "image" else c["text"])
    return "\n".join(parts)


# Monitor user-turn templates, rendered with placeholder inputs (2-frame window shown).
IMG = object()
identify_user = render(func("vlm_monitor_tune", "_identify_content")(
    [IMG, IMG], IMG, 2.0, wrist=[IMG, IMG]))
identify_user = identify_user.replace("2 overhead stills", "{N} overhead stills") \
    .replace("same 2 steps", "same {N} steps") \
    .replace("front frame 1/2\n", "front frame 1/{N}\n").replace("front frame 2/2", "...\nfront frame {N}/{N}") \
    .replace("wrist frame 1/2\n", "wrist frame 1/{N}\n").replace("wrist frame 2/2", "...\nwrist frame {N}/{N}")
compare_user_resolved = render(func("vlm_monitor_tune", "_compare_content")(
    "{sub_task}", {"reaching_for": "{reaching_for}", "held": "{held}", "candidates": []}))

P = lambda m, n: const(m, n)  # noqa: E731
# .format() templates: show single braces as the model sees them
F = lambda m, n: const(m, n).replace("{{", "{").replace("}}", "}")  # noqa: E731

# (section, [(title, role, source, text, note)])
SECTIONS = [
    ("Task decomposition (planner)", [
        ("Object identification (pass 1)", "system", "vlm.py: IDENTIFICATION_SYSTEM_PROMPT",
         P("vlm", "IDENTIFICATION_SYSTEM_PROMPT"),
         "User turn: [top-down camera] <image> List every visible object and surface."),
        ("Decomposition from objects (pass 2)", "system",
         "vlm.py: DECOMPOSITION_FROM_OBJECTS_SYSTEM_PROMPT",
         P("vlm", "DECOMPOSITION_FROM_OBJECTS_SYSTEM_PROMPT"),
         "User turn: [top-down camera] <image> Visible objects: [...] Instruction: {instruction}, "
         "followed on a retry round by the two notes below."),
        ("Retry-round context (appended to the pass-2 user turn)", "user",
         "vlm_robot_orchestrator.py: RETRY_DECOMPOSE_CONTEXT",
         P("vlm_robot_orchestrator", "RETRY_DECOMPOSE_CONTEXT"),
         "{completed} lists the sub-tasks already executed and verified."),
        ("Count-limited retry note (appended when the instruction names a count)", "user",
         "vlm_robot_orchestrator.py: COUNT_RETRY_NOTE",
         P("vlm_robot_orchestrator", "COUNT_RETRY_NOTE"), None),
    ]),
    ("Execution monitor (during VLA execution)", [
        ("Scene inventory (once per episode, initial frame)", "user",
         "vlm_monitor_tune.py: INVENTORY_PROMPT", P("vlm_monitor_tune", "INVENTORY_PROMPT"),
         "Sent with the initial front-camera image; no system prompt."),
        ("Sub-task object grounding", "user", "vlm_monitor_tune.py: MATCH_PROMPT",
         F("vlm_monitor_tune", "MATCH_PROMPT"),
         "Sent with the initial front-camera image; {choices} is the inventory names."),
        ("Reach check, pass 1: identify", "system",
         "vlm_monitor_tune.py: IDENTIFY_SYSTEM_PROMPT",
         P("vlm_monitor_tune", "IDENTIFY_SYSTEM_PROMPT"), None),
        ("Reach check, pass 1: user turn", "user", "vlm_monitor_tune.py: _identify_content",
         identify_user, "Each <image> is one camera still; {N} is the look-back window length. If the gripper "
         "already closed on an object earlier, a [HELD-OBJECT MEMORY ...] note naming it is "
         "inserted before the question."),
        ("Reach check, pass 2: compare", "system",
         "vlm_monitor_tune.py: COMPARE_SYSTEM_PROMPT",
         P("vlm_monitor_tune", "COMPARE_SYSTEM_PROMPT"),
         "Also the system prompt of the name-reconciliation call below."),
        ("Reach check, pass 2: user turn (text only)", "user",
         "vlm_monitor_tune.py: _compare_content", compare_user_resolved,
         "Shown for a resolved reach. When pass 1 answers 'unclear', the third line reads "
         "'The perception pass could not yet resolve which object the arm is going for "
         "(unclear).' and the candidates line lists pass 1's candidates."),
        ("Wrist verification probe (two fresh wrist frames at the grasp)", "user",
         "vlm_monitor_tune.py: HELD_PROBE_PROMPT", F("vlm_monitor_tune", "HELD_PROBE_PROMPT"),
         "Sent with one wrist image per call; {choices} is 'name (appearance), ..., "
         "other (an object not in this list)'."),
        ("Name reconciliation (reached name -> inventory entry)", "user",
         "vlm_monitor_tune.py: RECONCILE_PROMPT", F("vlm_monitor_tune", "RECONCILE_PROMPT"),
         "Text only, under COMPARE_SYSTEM_PROMPT; skipped when the names match as strings."),
    ]),
    ("Sub-task evaluation", [
        ("Sub-task success evaluator (video clip)", "system",
         "vlm_robot_orchestrator.py: EVALUATION_SYSTEM_PROMPT",
         P("vlm_robot_orchestrator", "EVALUATION_SYSTEM_PROMPT"),
         'User turn: 16-frame clip of the attempt, then: The robot just attempted this '
         'sub-task: "{sub_task}" Did it succeed?'),
        ("Vocabulary refinement (after a VOCAB failure)", "system",
         "vlm.py: REFINEMENT_SYSTEM_PROMPT", P("vlm", "REFINEMENT_SYSTEM_PROMPT"),
         'User turn: <image> The robot failed to execute: "{instruction}" Diagnosis: '
         '{reason} Suggest 3 alternative names for the target object.'),
    ]),
    ("Final whole-task evaluation (also the no-movement pre-check)", [
        ("BEFORE/AFTER scene survey", "system",
         "vlm.py: SCENE_CHANGE_SURVEY_SYSTEM_PROMPT",
         P("vlm", "SCENE_CHANGE_SURVEY_SYSTEM_PROMPT"),
         "User turn: [BEFORE — top-down camera, before the robot started] <image> "
         "[AFTER — top-down camera, now] <image> List every object, what it rested on in "
         "BEFORE, and what it rests on in AFTER."),
        ("Instruction requirements", "system",
         "vlm_robot_orchestrator.py: FINAL_REQUIREMENTS_SYSTEM_PROMPT",
         P("vlm_robot_orchestrator", "FINAL_REQUIREMENTS_SYSTEM_PROMPT"),
         "User turn: [BEFORE — the table before the robot started. Use it only to see what "
         "each listed object is (colour, material, kind); it does not show where anything "
         'is now.] <image> Objects on the table: [...] Instruction: "{instruction}". '
         "The pre-check passes the sub-task text as the instruction."),
    ]),
]


def tex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}"),
                 ("~", r"\textasciitilde{}")):
        s = s.replace(a, b) if a != "\\" else s.replace(a, "\x00")
    s = s.replace("\x00", r"\textbackslash{}")
    return s.replace("<image>", r"\texttt{<image>}").replace("->", r"$\rightarrow$")


tex = [r"""% Generated from data/vlm.py, data/vlm_robot_orchestrator.py, data/vlm_monitor_tune.py
% Requires: \usepackage{listings}
\lstdefinestyle{prompt}{basicstyle=\ttfamily\scriptsize, breaklines=true,
  breakatwhitespace=false, columns=fullflexible, keepspaces=true, frame=single,
  inputencoding=utf8, extendedchars=true,
  literate={—}{{---}}1 {–}{{--}}1 {’}{{'}}1 {→}{{$\rightarrow$}}1 {°}{{$^\circ$}}1}
\chapter{VLM Prompts}
\label{app:prompts}
All calls go to Qwen3-VL-8B-Instruct at temperature 0.1. Placeholders filled at run
time are shown in braces; \texttt{<image>} marks an image in the message.
"""]
md = ["# VLM prompts (full system)\n",
      "All calls: Qwen3-VL-8B-Instruct, temperature 0.1. `{...}` = filled at run time, "
      "`<image>` = an image in the message.\n"]
for sec, items in SECTIONS:
    tex.append(f"\n\\section{{{tex_escape(sec)}}}\n")
    md.append(f"\n## {sec}\n")
    for title, role, src, text, note in items:
        tex.append(f"\\subsection*{{{tex_escape(title)}}}\n"
                   f"\\noindent\\textit{{{role.capitalize()} message; source: "
                   f"\\texttt{{{tex_escape(src)}}}.}}"
                   + (f" {tex_escape(note)}" if note else "") + "\n")
        tex.append("\\begin{lstlisting}[style=prompt]\n" + text.strip("\n") + "\n\\end{lstlisting}\n")
        md.append(f"\n### {title}\n\n*{role.capitalize()} message — `{src}`.*"
                  + (f" {note}" if note else "") + f"\n\n```text\n{text.strip(chr(10))}\n```\n")

out.mkdir(parents=True, exist_ok=True)
(out / "appendix_prompts.tex").write_text("".join(tex))
(out / "appendix_prompts.md").write_text("".join(md))
print("wrote", out / "appendix_prompts.tex", out / "appendix_prompts.md")
