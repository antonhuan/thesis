"""Score _r2replay/results.json against ground truth read off the round-2 frames.
correct: the plan round 2 should produce (any of the listed sets); forbidden: objects
whose sub-task would break the prompt (excluded, not selected, over the count)."""
import json
import re

GT = {  # trial: (correct sets, forbidden objects)
    12: ([{"banana"}], set()), 14: ([set()], {"toy", "banana"}),
    16: ([set()], {"pouch"}), 17: ([set()], set()), 20: ([set()], {"pouch", "banana"}),
    24: ([{"pouch"}], set()), 28: ([set()], {"banana", "pouch"}), 29: ([set()], {"banana"}),
    34: ([set()], set()), 39: ([{"pouch"}, {"banana"}], set()),
    40: ([set()], {"toy", "pouch"}), 43: ([set()], {"toy", "banana"}),
    44: ([set()], {"toy"}), 45: ([{"banana"}], {"pouch"}), 46: ([set()], {"banana"}),
    47: ([{"pouch"}], set()), 50: ([set()], {"banana", "pouch"}), 53: ([{"toy"}], set()),
    55: ([set()], set()),
}
ALIAS = [("banana", "banana"), ("plush", "toy"), ("moose", "toy"), ("stuffed", "toy"),
         ("animal", "toy"), ("toy", "toy"), ("pouch", "pouch"), ("pencil", "pouch"),
         ("case", "pouch"), ("cylind", "pouch"), ("tube", "pouch"), ("roll", "pouch"),
         ("bag", "pouch"), ("chocolate", "pouch")]


def canon(st: str) -> str:
    m = re.match(r"put the (.+?) (?:on|in|into|onto) the", st.lower())
    name = m.group(1) if m else st.lower()
    return next((v for k, v in ALIAS if k in name), f"?{name}")


def judge(n, plan):
    if isinstance(plan, str):
        return "parse_fail", plan
    objs = [canon(p) for p in plan]
    correct, forbidden = GT[n]
    if n == 39 and {"pouch", "banana"} <= set(objs):
        return "BREAKS", objs
    if any(o in forbidden for o in objs):
        return "BREAKS", objs
    if any(o.startswith("?") for o in objs):
        return "invented", objs
    if set(objs) in correct:
        return "correct", objs
    return ("missed" if not objs or set(objs) < max(correct, key=len) else "redundant"), objs



res = json.load(open("_r2replay/results.json"))
tally = {v: {} for v in ("logged", "T", "F", "FN", "NF", "NT") if v in res[0]}
for r in res:
    row = []
    for v in tally:
        verdict, objs = judge(r["trial"], r[v])
        tally[v][verdict] = tally[v].get(verdict, 0) + 1
        row.append(f"{v}={verdict}{objs}")
    print(f"#{r['trial']:2d} " + " | ".join(row))
print()
for v, c in tally.items():
    print(f"{v:6} n={sum(c.values())}  " + "  ".join(f"{k}={c[k]}" for k in sorted(c)))
