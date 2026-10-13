"""MindGames diamond: muddy-children reasoning over several ignorance rounds, in matched pairs.

Each entailment needs at least two of its ignorance announcements ("nobody knows whether their own
forehead is muddy", "X does not know whether ..."): removing any one of them breaks it. The
"nobody knows" round may be repeated, as in the muddy-children puzzle, where a repeated round
carries new information. Each entailment has a twin non-entailment with the same sentences, reordered only
among announcements of the same kind (public announcements do not commute), so neither the words
nor the sequence of announcement kinds reveal the label. Hypotheses mostly ask whether someone
can know their own state, nested under 0-2 other people. Setups: forehead (everyone sees everyone
else) and explicit cards (each person sees each other card with probability 0.8); internal states
are left out (people know their own, so no ignorance round survives). 60 pairs per setup.

Labels come from mindgames.check and are checked against mindgames.pal (independent code) and
against the problem rebuilt from the English text (mindgames.english).

    python -m mindgames.diamond --out release/hf/diamond/test.parquet
"""
import argparse
import hashlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from inflect import engine

from mindgames.check import check

POOLS = {"forehead": [(20261013, 1_000_000), (30_000_000, 5_000_000)], "explicit": [(60_000_000, 3_000_000)]}
PAIRS_PER_SETUP = 60
SALT = "mindgames-diamond-2"
VISIBILITY = 0.8
NAMES = ["Alice", "Bruno", "Chloe", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jonas", "Keiko", "Liam",
         "Maya", "Nadia", "Omar", "Priya", "Quentin", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wen", "Yusuf", "Zoe"]
LETTERS = "abcdefghij"
QUANTIFIERS = ["someone", "everyone", "not everyone", "nobody"]
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
            rng.random((n, n))  # discarded draw, kept so that the published items rebuild exactly
            shown = rng.random((n, n)) < VISIBILITY
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


def support_ratio(head, anns, hyp):
    """Share of the remaining worlds where the hypothesis holds (1.0 for entailments)."""
    n_all = check(problem(head, anns, "Top"))[1]
    n_true = check(problem(head, [*anns, hyp], "Top"))[1]  # announcing it keeps the worlds where it holds
    return n_true / n_all


def kind(english):
    if english.startswith("nobody knows"):
        return "nobody knows"
    if " does not know " in english:
        return "does not know"
    return "knows" if " knows " in english else "fact"


def announcement(item, rng):
    r = rng.random()
    if r < 0.4:
        return item.nobody_knows()
    if r < 0.7:
        return item.knowledge(0, kind="whether", negated=True)
    if r < 0.85:
        return item.knowledge(0, kind="whether")
    return item.atom(int(rng.integers(item.n)), negated=bool(rng.random() < 0.5))


def announcements(item, rng, k):
    out = [item.quantifier("someone")] if rng.random() < 0.8 else [item.quantifier("not everyone")]
    while len(out) < k:
        a = announcement(item, rng)
        if a[0].startswith("(~(Agent") or a not in out:  # "nobody knows" rounds may repeat
            out.append(a)
    return out


def hypothesis(item, rng, depth):
    """Mostly "A can now know that B can know whether B's ..."; sometimes any proposition."""
    if rng.random() < 0.3:
        return item.knowledge(depth, verb="can now know", inner_verb="can know")
    agents = [int(rng.integers(item.n))]
    while len(agents) <= depth:
        agents.append(int(rng.choice([a for a in range(item.n) if a != agents[-1]])))
    formal, english = item.atom(agents[-1])
    for k, a in enumerate(reversed(agents)):
        how = "whether" if k == 0 else str(rng.choice(["that", "whether"]))
        verb = "can now know" if k == len(agents) - 1 else "can know"
        inner = f"({formal})" if k else formal
        formal, english = f"(Agent{LETTERS[a]} knows {how} {inner})", f"{item.names[a]} {verb} {how} {english}"
    return formal, english


def candidate(seed, setup):
    rng = np.random.default_rng(seed)
    item = Item(setup, int(rng.integers(5, 9)), rng)
    obs, obs_text = item.observations()
    head = f"VARS {','.join(str(j + 1) for j in range(item.n))} LAW Top {obs}"
    anns = announcements(item, rng, int(rng.integers(4, 8)))
    depth = int(rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3]))
    hyp, hyp_text = hypothesis(item, rng, depth)
    F = [a for a, _ in anns]
    label, worlds = check(problem(head, F, hyp))
    if not label or worlds < 2:
        return None
    needed = [i for i, a in enumerate(F) if "~(Agent" in a and not check(problem(head, F[:i] + F[i + 1:], hyp))[0]]
    if len(needed) < 2:
        return None
    kinds = [kind(e) for _, e in anns]
    if all(len({F[j] for j in range(len(F)) if kinds[j] == t}) < 2 for t in set(kinds)):
        return None
    for _ in range(40):
        perm = list(range(len(F)))
        for t in sorted(set(kinds)):
            pos = [j for j, x in enumerate(kinds) if x == t]
            for a, b in zip(pos, rng.permutation(pos)):
                perm[a] = int(b)
        G = [F[j] for j in perm]
        if G == F:
            continue
        label2, worlds2 = check(problem(head, G, hyp))
        if not label2 and worlds2 >= 2:
            twin = [anns[j] for j in perm]
            break
    else:
        return None

    def row(A, label, worlds, twin_id):
        formal = [a for a, _ in A]
        return {"premise": obs_text + "".join(f" It is publicly announced that {e}." for _, e in A),
                "hypothesis": hyp_text + ".", "label": label, "smcdel_problem": problem(head, formal, hyp),
                "setup": setup, "n_agents": item.n, "n_announcements": len(A), "hypothesis_depth": depth,
                "names": item.names, "index": 2 * (3_000_000 + seed) + twin_id, "pair": 3_000_000 + seed,
                "worlds_left": worlds, "needed_ignorance": len(needed) if twin_id == 0 else 0,
                "support": support_ratio(head, formal, hyp)}
    return [row(anns, "entailment", worlds, 0), row(twin, "not_entailment", worlds2, 1)]


def _forehead(seed):
    return candidate(seed, "forehead") if seed % 3 == 0 else None


def _explicit(seed):
    return candidate(seed, "explicit")


def build(pairs_per_setup=PAIRS_PER_SETUP):
    parts = []
    with Pool() as workers:
        for setup, fn in (("forehead", _forehead), ("explicit", _explicit)):
            seeds = [s for start, n in POOLS[setup] for s in range(start, start + n)]
            rows = [r for c in workers.imap(fn, seeds, chunksize=500) if c for r in c]
            df = pd.DataFrame(rows)
            pairs = sorted(df.pair.unique(), key=lambda p: hashlib.sha256(f"{SALT}:{p}".encode()).hexdigest())
            parts.append(df[df.pair.isin(pairs[:pairs_per_setup])])
    out = pd.concat(parts)
    order = out["index"].map(lambda i: hashlib.sha256(f"{SALT}:{i}".encode()).hexdigest())
    return out.assign(_o=order).sort_values("_o").drop(columns="_o").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release/hf/diamond/test.parquet")
    ap.add_argument("--pairs-per-setup", type=int, default=PAIRS_PER_SETUP)
    args = ap.parse_args()
    out = build(args.pairs_per_setup)
    out.to_parquet(args.out, index=False)
    print(len(out), "items", out.groupby(["setup", "label"]).size().to_dict())


if __name__ == "__main__":
    main()
