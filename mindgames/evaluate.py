"""Evaluate models on MindGames through OpenRouter, then score.

    export OPENROUTER_API_KEY=...
    python -m mindgames.evaluate run openai/gpt-6-luna-20260922 --limit 400
    python -m mindgames.evaluate score

Replies go to results/<config>/<model>.jsonl; a rerun only calls the items still missing.
"""
import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

REPO = "sileod/mindgames"
REVISION = "v1.1"
PROMPT = """Situation: {premise}

Assume everyone reasons perfectly, public announcements are true and heard by everyone, and each person knows only what they observe or can deduce.

Statement: {hypothesis}

Question: Does the statement follow from the situation?
Answer only Yes or No."""

ANSWER = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
FINAL_ANSWER = re.compile(r"(?:final answer|answer)\s*:?\s*(yes|no)\b(?!.*(?:final answer|answer)\s*:?\s*(?:yes|no)\b)",
                          re.IGNORECASE | re.DOTALL)


def parse(text):
    """Yes/No from a reply: a final \\boxed{}, a leading or "Answer:" Yes/No, else a bare Yes/No line."""
    if not isinstance(text, str) or not text.strip():
        return None
    boxed = re.findall(r"\\boxed\{\s*(?:\\text(?:bf)?\{)?\s*(yes|no)\s*\}?\s*\}", text, re.IGNORECASE)
    if boxed:
        return boxed[-1].lower() == "yes"
    clean = re.sub(r"[*_`#]", "", text).strip()
    match = ANSWER.match(clean) or FINAL_ANSWER.search(clean)
    if match:
        return match.group(1).lower() == "yes"
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    for line in (lines[-1], lines[0]) if lines else ():
        match = re.fullmatch(r"(?:final answer|answer)?\s*:?\s*(yes|no)[.!]?", line, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "yes"
    return None


def load(config, split="test", data_dir=None):
    """A config of the HF dataset, or the same layout from a local folder built by mindgames.release."""
    if data_dir:
        return pd.read_parquet(Path(data_dir) / ("data" if config == "default" else config) / f"{split}.parquet")
    return load_dataset(REPO, config, split=split, revision=REVISION).to_pandas()


def slug(model, effort="medium"):
    """File name of a run; a non-default reasoning effort is its own entry (<model>@<effort>)."""
    return model.split("/")[-1] + ("" if effort == "medium" else f"@{effort}")


async def run(model, config, effort, max_tokens, concurrency, limit, data_dir=None):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    df = load(config, data_dir=data_dir)
    if limit:
        df = df.head(limit)
    path = Path("results") / config / f"{slug(model, effort)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {r["index"] for r in map(json.loads, path.open()) if not r.get("error")}
    todo = df[~df["index"].isin(done)]
    sem = asyncio.Semaphore(concurrency)
    print(f"{model}: {len(todo)} of {len(df)} items to run")

    async def one(row):
        async with sem:
            record = {"index": int(row["index"]), "model": model, "label": row.label, "reasoning_effort": effort,
                      "max_tokens": max_tokens}
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": PROMPT.format(premise=row.premise, hypothesis=row.hypothesis)}],
                    extra_body={"reasoning": {"effort": effort}, "usage": {"include": True}}, timeout=600)
                usage = r.usage.model_dump() if r.usage else {}
                message = r.choices[0].message
                record.update(reply=message.content, reasoning=getattr(message, "reasoning", None), finish_reason=r.choices[0].finish_reason,
                              provider_model=r.model, usage=usage, cost_usd=usage.get("cost"))
            except Exception as e:  # noqa: BLE001 - stored so that a rerun retries it
                record["error"] = f"{type(e).__name__}: {e}"[:500]
            return record

    with path.open("a") as f:
        for coro in asyncio.as_completed([one(row) for _, row in todo.iterrows()]):
            f.write(json.dumps(await coro) + "\n")
            f.flush()


def read(path):
    df = pd.read_json(path, lines=True)
    if "error" in df:
        df = df[df.error.isna()]
    return df.drop_duplicates("index", keep="last")


def score(config, n_boot=10000, seed=0):
    rows = []
    for path in sorted((Path("results") / config).glob("*.jsonl")):
        df = read(path)
        pred = df.reply.map(parse)
        correct = (pred == (df.label == "entailment")).to_numpy(float)
        boot = np.random.default_rng(seed).choice(correct, (n_boot, len(correct))).mean(1)
        rows.append({"model": path.stem, "n": len(df), "accuracy": 100 * correct.mean(),
                     "ci_low": 100 * np.quantile(boot, 0.025), "ci_high": 100 * np.quantile(boot, 0.975),
                     "unparsed": int(pred.isna().sum()), "truncated": int((df.finish_reason == "length").sum()),
                     "cost_usd": df.cost_usd.astype(float).sum()})
    table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    print(table.to_string(index=False, float_format="%.1f"))
    return table


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("models", nargs="+")
    r.add_argument("--config", default="default")
    r.add_argument("--reasoning-effort", default="medium")
    r.add_argument("--max-tokens", type=int, default=8192)
    r.add_argument("--concurrency", type=int, default=32)
    r.add_argument("--limit", type=int)
    r.add_argument("--data-dir", help="local release folder instead of the HF dataset")
    s = sub.add_parser("score")
    s.add_argument("--config", default="default")
    args = ap.parse_args()
    if args.cmd == "run":
        async def all_models():
            await asyncio.gather(*(run(m, args.config, args.reasoning_effort, args.max_tokens, args.concurrency,
                                       args.limit, args.data_dir) for m in args.models))
        asyncio.run(all_models())
    else:
        score(args.config)


if __name__ == "__main__":
    main()
