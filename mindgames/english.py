"""Rebuild an item's formal problem from its English premise and hypothesis alone.

An independent check of the generators: labelling the rebuilt problem with mindgames.check
must give the item's label (it does for every 1.1 test item and every diamond item).
"""
import re

L = "abcdefghij"

def parse_item(premise, hypothesis, names):
    """Formal problem string for an item; ``names`` gives the agent order."""
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    sents = [s.strip() for s in re.split(r"(?<=\.)\s+", premise.strip()) if s.strip()]
    if any("mirror" in s for s in sents): setup = "forehead_mirror"
    elif any("thirsty" in s for s in sents): setup = "internal"
    elif any("card" in s for s in sents): setup = "explicit"
    else: setup = "forehead"
    if setup == "forehead": obs = {i: [j for j in range(n) if j != i] for i in range(n)}
    elif setup == "internal": obs = {i: [i] for i in range(n)}
    elif setup == "forehead_mirror": obs = {i: list(range(n)) for i in range(n)}
    else:
        obs = {i: [] for i in range(n)}
        for o, v in re.findall(r"(\w+)'s card is shown to (\w+)\.", premise): obs[idx[v]].append(idx[o])
    ids = [str(j + 1) for j in range(n)]
    def prop(p):
        p = p.strip().rstrip(".")
        p = p.removeprefix("or not ")
        for q, f in [("not everyone", f"(~({'&'.join(ids)}))"), ("nobody", f"(~({'|'.join(ids)}))"),
                     ("someone", f"({'|'.join(ids)})"), ("everyone", f"({'&'.join(ids)})")]:
            if re.fullmatch(q + r"(?:'s forehead is muddy| is thirsty| picked a red card)", p): return f
        m = re.fullmatch(r"(\w+)(?:'s forehead is (not )?muddy| is (not )?thirsty| picked a (red|black) card)", p)
        assert m, p
        neg = bool(m.group(2) or m.group(3) or m.group(4) == "black")
        v = str(idx[m.group(1)] + 1)
        return f"~({v})" if neg else v
    def know(p):
        m = re.match(r"(\w+) (knows|does not know|can now know|can know|cannot know) (that|whether) (.*)$", p)
        if not m or m.group(1) not in idx: return prop(p)
        inner = know(m.group(4))
        f = f"(Agent{L[idx[m.group(1)]]} knows {m.group(3)} ({inner}))"
        return f"~{f}" if "not" in m.group(2) else f
    anns = []
    for s in sents:
        m = re.match(r"It is publicly announced that (.*)\.$", s)
        if not m: continue
        a = m.group(1)
        if a.startswith("nobody knows whether"):
            anns.append("(" + " & ".join(f"~(Agent{L[i]} knows whether {i+1})" for i in range(n)) + ")")
        else:
            anns.append(know(a))
    head = f"VARS {','.join(ids)} LAW Top OBS " + " ".join(f"Agent{L[i]}:" + ",".join(str(j+1) for j in sorted(v)) for i, v in obs.items() if v)
    return f"{head} VALID? " + " ".join(f"[ ! {a} ]" for a in anns) + " " + know(hypothesis.rstrip("."))
