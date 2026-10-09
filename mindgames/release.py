"""Build MindGames 1.1 from the 1.0 release.

1.0 labelled each item with SMCDEL from its ``smcdel_problem``. In the ``explicit`` setup,
the observations written in ``smcdel_problem`` were shifted to the wrong agent whenever an
agent observed nothing, so the English premise and the labelled problem disagreed. 1.1
rebuilds the observations from the English premise, relabels every item with
``mindgames.check``, and drops items whose premise becomes contradictory.

    python -m mindgames.release --out release/hf
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from mindgames.check import check

REPO = "sileod/mindgames"
V1_0 = "e378f4c70f66413a00c664373f4f43064b3ba752"
FILES = {
    "train": "data/train-00000-of-00001-29e951c428782278.parquet",
    "validation": "data/validation-00000-of-00001-c11c488b5d7e6ebd.parquet",
    "test": "data/test-00000-of-00001-7dfe9e22268ffc8b.parquet",
}
LABELS = {1: "entailment", 0: "not_entailment"}


def load_v1_0():
    return {split: pd.read_parquet(hf_hub_download(REPO, path, repo_type="dataset", revision=V1_0))
            for split, path in FILES.items()}


def text_observations(row):
    """OBS clause of an explicit-setup item, read from its English premise."""
    names = list(row.names)
    seen = {i: [] for i in range(row.n_agents)}
    for owner, viewer in re.findall(r"(\w+)'s card is shown to (\w+)\.", row.premise):
        seen[names.index(viewer)].append(names.index(owner) + 1)
    return "OBS " + " ".join(f"Agent{'abcdefghij'[i]}:" + ",".join(map(str, sorted(v)))
                             for i, v in seen.items() if v) + " VALID?"


def with_obs(problem, obs):
    return re.sub(r"OBS .*? VALID\?", obs, problem)


def fix(df):
    """Return the fixed split and a change log (one dict per changed or dropped item)."""
    df = df.copy()
    log = []
    keep = []
    for i, row in df.iterrows():
        if row.setup == "explicit":
            obs = text_observations(row)
            df.at[i, "smcdel_problem"] = with_obs(row.smcdel_problem, obs)
            df.at[i, "pbcheck"] = with_obs(row.pbcheck, obs)
        if check(df.at[i, "pbcheck"])[1] == 0:
            log.append({"index": int(row["index"]), "change": "dropped (premise contradictory)"})
            keep.append(False)
            continue
        label = LABELS[check(df.at[i, "smcdel_problem"])[0]]
        if label != row.label:
            log.append({"index": int(row["index"]), "change": f"label {row.label} -> {label}"})
            df.at[i, "label"] = label
        keep.append(True)
    df = df[keep].reset_index(drop=True)
    df["s-l"] = df.setup + "-" + (df.label == "entailment").astype(int).astype(str)
    agree = df.deberta_pred == (df.label == "entailment").astype(int)
    df["difficulty"] = (1 - df.deberta_confidence).where(agree, 1 + df.deberta_confidence)
    return df, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release/hf")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    changes = {}
    fixed = {}
    for split, df in load_v1_0().items():
        fixed[split], changes[split] = fix(df)
        fixed[split].to_parquet(out / "data" / f"{split}.parquet", index=False)
        print(split, len(df), "->", len(fixed[split]),
              {c: sum(x["change"].startswith(c) for x in changes[split]) for c in ("label", "dropped")})
    (out / "changes-1.1.json").write_text(json.dumps(changes, indent=1))


if __name__ == "__main__":
    main()
