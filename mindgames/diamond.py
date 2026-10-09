"""MindGames diamond: newly generated, harder problems in the MindGames templates.

Compared with the main splits: 5-6 people (with cards, each sees a random subset of the
others' cards), 4-6 distinct public announcements, mostly ignorance rounds ("X does not know
whether ...", "nobody knows whether their own card is red"), and hypotheses about 2-3 nested
agents. At least two worlds remain after the announcements; entailments stop following when
some ignorance round is removed; non-entailments hold in at least half of the remaining worlds,
or would have followed without one of the later announcements. Internal-state premises say
that people know their own state; the mirror setup is left out (everyone sees everything).
Labels come from mindgames.check; 40 items per setup x label, in SHA-256 order.

    python -m mindgames.diamond --out release/hf/diamond/test.parquet
"""
import argparse
import hashlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from inflect import engine

from mindgames.check import check

NAMES = ["Alice", "Bruno", "Chloe", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jonas", "Keiko", "Liam",
         "Maya", "Nadia", "Omar", "Priya", "Quentin", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wen", "Yusuf", "Zoe"]
LETTERS = "abcdefghij"
SETUPS = ["forehead", "internal", "explicit"]  # with a mirror everyone sees everything: no hard items
QUANTIFIERS = ["someone", "everyone", "not everyone", "nobody"]
SEED = 20261009
words = engine().number_to_words


class Item:
    """One problem: the formal SMCDEL-style string and its English, built together."""

    def __init__(self, setup, n, rng):
        self.setup, self.n, self.rng = setup, n, rng
        self.names = list(rng.choice(NAMES, n, replace=False))
        if setup == "forehead":
            self.obs = {i: [j for j in range(n) if j != i] for i in range(n)}
        elif setup == "forehead_mirror":
            self.obs = {i: list(range(n)) for i in range(n)}
        elif setup == "internal":
            self.obs = {i: [i] for i in range(n)}
        else:
            shown = rng.random((n, n)) < 0.35
            np.fill_diagonal(shown, False)
            self.obs = {i: list(np.flatnonzero(shown[i])) for i in range(n)}

    # atoms -------------------------------------------------------------------------------
    def atom(self, j, negated=False):
        name = self.names[j]
        english = {"internal": f"{name} is {'not ' if negated else ''}thirsty",
                   "explicit": f"{name} picked a {'black' if negated else 'red'} card"}.get(
            self.setup, f"{name}'s forehead is {'not ' if negated else ''}muddy")
        return (f"~({j + 1})" if negated else f"{j + 1}"), english

    def quantifier(self, q):
        ids = [str(j + 1) for j in range(self.n)]
        formal = {"someone": f"({'|'.join(ids)})", "everyone": f"({'&'.join(ids)})",
                  "not everyone": f"(~({'&'.join(ids)}))", "nobody": f"(~({'|'.join(ids)}))"}[q]
        subject = {"internal": f"{q} is thirsty", "explicit": f"{q} picked a red card"}.get(
            self.setup, f"{q}'s forehead is muddy")
        return formal, subject

    def proposition(self):
        if self.rng.random() < 0.4:
            return self.quantifier(self.rng.choice(QUANTIFIERS))
        return self.atom(int(self.rng.integers(self.n)), negated=bool(self.rng.random() < 0.3))

    # knowledge ---------------------------------------------------------------------------
    def knowledge(self, depth, kind=None, negated=False, exclude=None, verb="knows", inner_verb="knows"):
        agent = int(self.rng.choice([a for a in range(self.n) if a != exclude]))
        kind = kind or self.rng.choice(["that", "whether"])
        if depth == 0:
            formal_sub, english_sub = self.proposition()
        else:
            formal_sub, english_sub = self.knowledge(depth - 1, exclude=agent, verb=inner_verb, inner_verb=inner_verb)
            formal_sub = f"({formal_sub})"
        if kind == "whether" and english_sub.split()[0] in ("everyone", "nobody"):
            english_sub = "or not " + english_sub
        formal = f"Agent{LETTERS[agent]} knows {kind} {formal_sub}"
        verb = {"knows": "does not know", "can now know": "cannot know"}[verb] if negated else verb
        return (f"~({formal})" if negated else f"({formal})"), f"{self.names[agent]} {verb} {kind} {english_sub}"

    def nobody_knows(self):
        """Everyone simultaneously says they cannot tell their own state (a muddy-children round)."""
        formal = "(" + " & ".join(f"~(Agent{LETTERS[i]} knows whether {i + 1})" for i in range(self.n)) + ")"
        own = {"internal": "they are thirsty", "explicit": "their own card is red"}.get(
            self.setup, "their own forehead is muddy")
        return formal, f"nobody knows whether {own}"

    def announcements(self, k):
        """'someone ...' first, then distinct announcements: mostly ignorance rounds, some knowledge,
        a few single facts (never quantified facts, which would settle everything at once)."""
        out = [self.quantifier("someone")]
        for _ in range(200):
            if len(out) == k:
                break
            r = self.rng.random()
            if r < 0.35:
                new = self.nobody_knows()
                if out[-1] == new:
                    continue
                out.append(new)
                continue
            if r < 0.6:  # the muddy-children step
                new = self.knowledge(0, kind="whether", negated=True)
            elif r < 0.8:
                new = self.knowledge(0, kind="whether")
            else:
                new = self.atom(int(self.rng.integers(self.n)), negated=bool(self.rng.random() < 0.5))
            if new not in out:
                out.append(new)
        return out

    # rendering ---------------------------------------------------------------------------
    def observations(self):
        formal = "OBS " + " ".join(f"Agent{LETTERS[i]}:" + ",".join(str(j + 1) for j in v)
                                   for i, v in self.obs.items() if v)
        text = f"There are {words(self.n)} persons. Everyone is visible to others."
        if self.setup == "forehead_mirror":
            text += " There is a mirror in the room."
        elif self.setup == "internal":
            text += " Each person knows whether they are thirsty."
        elif self.setup == "explicit":
            text += " Each person draws a card, face unrevealed (red or black)."
            text += "".join(f" {self.names[j]}'s card is shown to {self.names[i]}."
                            for i, v in self.obs.items() for j in v)
        return formal, text


def problem(head, anns, hyp):
    return f"{head} VALID? " + " ".join(f"[ ! {a} ]" for a in anns) + f" {hyp}"


def candidate(seed):
    rng = np.random.default_rng(seed)
    setup = SETUPS[seed % len(SETUPS)]
    item = Item(setup, int(rng.integers(5, 7)), rng)
    obs, obs_text = item.observations()
    head = f"VARS {','.join(str(j + 1) for j in range(item.n))} LAW Top {obs}"
    anns = item.announcements(int(rng.integers(4, 7)))
    depth = int(rng.choice([1, 2], p=[0.4, 0.6]))
    hyp, hyp_text = item.knowledge(depth, verb="can now know", inner_verb="can know")
    formal = problem(head, [a for a, _ in anns], hyp)
    label, survivors = check(formal)
    if survivors == 0:
        return None
    formal_anns = [a for a, _ in anns]
    # ignorance rounds without which the hypothesis no longer follows
    needed = sum(check(problem(head, formal_anns[:i] + formal_anns[i + 1:], hyp))[0] == 0
                 for i, a in enumerate(formal_anns) if label and "~(Agent" in a)
    support = support_ratio(head, formal_anns, hyp)
    # announcements without which the (non-)hypothesis would have followed: more information, less knowledge
    trap = (not label) and any(check(problem(head, formal_anns[:i] + formal_anns[i + 1:], hyp))[0]
                               for i in range(1, len(formal_anns)))
    premise = obs_text + "".join(f" It is publicly announced that {e}." for _, e in anns)
    return {"premise": premise, "hypothesis": hyp_text + ".", "label": "entailment" if label else "not_entailment",
            "smcdel_problem": formal, "setup": setup, "n_agents": item.n, "n_announcements": len(anns),
            "hypothesis_depth": depth, "names": item.names, "index": 1_000_000 + seed,
            "worlds_left": survivors, "needed_ignorance": int(needed), "support": support,
            "trap": bool(trap)}


def support_ratio(head, anns, hyp):
    """Share of the remaining worlds where the hypothesis holds (1.0 for entailments)."""
    n_all = check(problem(head, anns, "Top"))[1]
    n_true = check(problem(head, [*anns, hyp], "Top"))[1]  # announcing it keeps the worlds where it holds
    return n_true / n_all


def build(per_cell=40, pool=300000, seed=SEED):
    with Pool() as workers:
        rows = [c for c in workers.map(candidate, range(seed, seed + pool), chunksize=500) if c]
    df = pd.DataFrame(rows)
    entail = (df.label == "entailment") & (df.needed_ignorance >= 1)
    tempting = (df.label == "not_entailment") & ((df.support >= 0.5) | df.trap)
    hard = df[(df.worlds_left >= 2) & (entail | tempting)].copy()
    hard["_order"] = hard["index"].map(lambda i: hashlib.sha256(f"mindgames-diamond:{i}".encode()).hexdigest())
    cells = [g.sort_values("_order").head(per_cell) for _, g in hard.groupby(["setup", "label"])]
    assert all(len(c) == per_cell for c in cells), hard.groupby(["setup", "label"]).size()
    return pd.concat(cells).sort_values("_order").drop(columns="_order").reset_index(drop=True), df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release/hf/diamond/test.parquet")
    ap.add_argument("--per-cell", type=int, default=40)
    ap.add_argument("--pool", type=int, default=300000)
    args = ap.parse_args()
    out, pool = build(args.per_cell, args.pool)
    out.to_parquet(args.out, index=False)
    print(len(pool), "candidates;", len(out), "selected", out.groupby(["setup", "label"]).size().to_dict())


if __name__ == "__main__":
    main()
