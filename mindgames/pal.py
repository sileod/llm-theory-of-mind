"""A second, independently written public-announcement-logic checker (no code shared with
mindgames.check); label(problem) -> (valid, worlds left). Used to cross-check diamond labels.

    python -m mindgames.pal release/hf/diamond/test.parquet
"""
import itertools
import re
import sys

import pandas as pd


def tokenize(s):
    return re.findall(r"Agent[a-j]|knows|whether|that|Top|\d+|[()~&|]", s)

def parse(toks):
    def expr():
        left = unary()
        while toks and toks[0] in "&|":
            op = toks.pop(0); right = unary(); left = (op, left, right)
        return left
    def unary():
        t = toks.pop(0)
        if t == "~": return ("~", unary())
        if t == "(":
            e = expr(); assert toks.pop(0) == ")"; return e
        if t.startswith("Agent"):
            assert toks.pop(0) == "knows"; kind = toks.pop(0); return ("K", "abcdefghij".index(t[-1]), kind, unary())
        if t == "Top": return ("T",)
        return ("p", int(t) - 1)
    return expr()

def holds(f, w, W, obs):
    t = f[0]
    if t == "T": return True
    if t == "p": return w[f[1]]
    if t == "~": return not holds(f[1], w, W, obs)
    if t == "&": return holds(f[1], w, W, obs) and holds(f[2], w, W, obs)
    if t == "|": return holds(f[1], w, W, obs) or holds(f[2], w, W, obs)
    _, a, kind, g = f
    acc = [v for v in W if all(v[j] == w[j] for j in obs[a])]
    vals = {holds(g, v, W, obs) for v in acc}
    return vals == {True} if kind == "that" else len(vals) == 1

def label(problem):
    head, rest = problem.split(" VALID? ")
    n = len(head.split("VARS ")[1].split(" LAW")[0].split(","))
    obs = {i: [] for i in range(n)}
    for a, v in re.findall(r"Agent([a-j]):([\d,]+)", head):
        obs["abcdefghij".index(a)] = [int(x) - 1 for x in v.split(",")]
    anns = re.findall(r"\[ ! (.*?) \]", rest)
    hyp = rest.rsplit("] ", 1)[1] if anns else rest
    W = list(itertools.product([False, True], repeat=n))
    for a in anns:
        f = parse(tokenize(a)); W = [w for w in W if holds(f, w, W, obs)]
    h = parse(tokenize(hyp))
    return all(holds(h, w, W, obs) for w in W), len(W)

if __name__ == "__main__":
    d = pd.read_parquet(sys.argv[1])
    res = [label(p) for p in d.smcdel_problem]
    bad = sum(l != (y == "entailment") or s != wl for (l, s), y, wl in zip(res, d.label, d.worlds_left))
    print("independent checker disagreements (label or world count):", bad, "of", len(d))
