"""Explicit-state model checker for MindGames problems (public announcement logic).

Replaces the SMCDEL web service used to label MindGames 1.0. A problem string looks like
``VARS 1,2 LAW Top OBS Agenta:2 Agentb:1 VALID? [ ! (1|2) ] (Agenta knows whether 2)``:
worlds are all valuations of VARS, an agent cannot tell apart worlds that agree on the
variables it observes, each announcement removes the worlds where it is false, and the
problem is valid when the final formula holds in every remaining world.
"""
import re
from itertools import product

TOK = re.compile(r"\s*(knows whether|knows that|VALID\?|Agent[a-z]|Top|\d+|[~&|()\[\]!])")

def tokenize(s):
    out, i = [], 0
    s = s.strip()
    while i < len(s):
        m = TOK.match(s, i)
        if not m:
            raise ValueError(f"bad token at {s[i:i+20]!r}")
        out.append(m.group(1)); i = m.end()
    return out

class P:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else None
    def eat(self, x=None):
        tok = self.t[self.i]
        if x is not None and tok != x: raise ValueError(f"expected {x} got {tok}")
        self.i += 1; return tok
    def expr(self):
        node = self.unary()
        while self.peek() in ("&", "|"):
            op = self.eat(); node = (op, node, self.unary())
        return node
    def unary(self):
        tok = self.peek()
        if tok == "~": self.eat(); return ("not", self.unary())
        if tok == "(":
            self.eat(); e = self.expr(); self.eat(")"); return e
        if tok == "Top": self.eat(); return ("top",)
        if tok.isdigit(): self.eat(); return ("var", int(tok))
        if tok.startswith("Agent"):
            ag = self.eat(); kind = self.eat(); return ("K" if kind == "knows that" else "Kw", ag, self.unary())
        raise ValueError(f"unexpected {tok}")

def parse_problem(s):
    m = re.match(r"VARS ([\d,]+) LAW (\S+) OBS (.*?) VALID\? (.*)$", s.strip())
    vars_ = [int(v) for v in m.group(1).split(",")]
    obs = {}
    for part in m.group(3).split():
        a, vs = part.split(":")
        obs[a] = [int(v) for v in vs.split(",") if v]
    p = P(tokenize(m.group(4)))
    anns = []
    while p.peek() == "[":
        p.eat("["); p.eat("!"); anns.append(p.expr()); p.eat("]")
    hyp = p.expr()
    assert p.peek() is None, "trailing tokens"
    assert m.group(2) == "Top"
    return vars_, obs, anns, hyp

def check(s):
    """Return (1 if valid else 0, number of worlds left after the announcements)."""
    vars_, obs, anns, hyp = parse_problem(s)
    worlds = [dict(zip(vars_, bits)) for bits in product([0, 1], repeat=len(vars_))]
    M = set(range(len(worlds)))
    def view(a, w):  # agents absent from OBS observe nothing
        return tuple(worlds[w][v] for v in obs.get(a, []))
    def ev(f, M):
        k = f[0]
        if k == "top": return set(M)
        if k == "var": return {w for w in M if worlds[w][f[1]]}
        if k == "not": return M - ev(f[1], M)
        if k == "&": return ev(f[1], M) & ev(f[2], M)
        if k == "|": return ev(f[1], M) | ev(f[2], M)
        a, sub = f[1], ev(f[2], M)
        cls = {}
        for w in M: cls.setdefault(view(a, w), set()).add(w)
        out = set()
        for c in cls.values():
            if k == "K" and c <= sub: out |= c
            if k == "Kw" and (c <= sub or not (c & sub)): out |= c
        return out
    for a in anns:
        M = ev(a, M)
    return int(ev(hyp, M) == M), len(M)
