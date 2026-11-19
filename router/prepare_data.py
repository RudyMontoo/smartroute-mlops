"""Build the query-difficulty dataset from real public sources.

The label of each query is the dataset it came from. This matters: if we generated
queries from templates ("Explain {topic}" -> moderate), the classifier would just
relearn our own templates, report a meaningless ~99% accuracy, and tell us nothing
about real traffic. Grounding the labels in independent datasets means the model has
to learn something real, and an 80-88% macro-F1 is an honest number.

    simple    SQuAD v2      short factual lookups; a small model answers these fine
    moderate  Dolly 15k     open-ended explanation/brainstorming; needs synthesis
    complex   GSM8K + MBPP  multi-step math reasoning and code generation

Downloads are cached in data/raw/ so reruns are offline and fast.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data"

SEED = 42
PER_CLASS = 300
HOLDOUT_SIZE = 200

SOURCES = {
    "squad": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "mbpp": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
    "dolly": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
}

# Dolly categories that genuinely need synthesis rather than a lookup or a proof.
DOLLY_CATEGORIES = {"open_qa", "general_qa", "brainstorming"}

MIN_CHARS = 15
MAX_CHARS = 600


def download(name: str, url: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{name}{Path(url).suffix or '.json'}"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {name}: cached ({dest.stat().st_size // 1024} KB)")
        return dest
    print(f"  {name}: downloading ...")
    req = urllib.request.Request(url, headers={"User-Agent": "smartroute-dataprep/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
        fh.write(resp.read())
    print(f"  {name}: {dest.stat().st_size // 1024} KB")
    return dest


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def load_squad(path: Path) -> list[dict]:
    """Short factual questions. Keep only answerable ones so eval has ground truth."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for article in payload["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                if qa.get("is_impossible") or not qa.get("answers"):
                    continue
                q = clean(qa["question"])
                # Long SQuAD questions tend to be context-dependent and read as
                # moderate, which would blur the class boundary. Keep them crisp.
                if not (MIN_CHARS <= len(q) <= 120):
                    continue
                rows.append(
                    {
                        "query": q,
                        "label": "simple",
                        "source": "squad_v2",
                        "answer": clean(qa["answers"][0]["text"]),
                        "answer_type": "span",
                    }
                )
    return rows


def load_dolly(path: Path) -> list[dict]:
    """Open-ended instructions. Drop anything with a context blob attached: those are
    reading-comprehension tasks in disguise, not standalone queries."""
    rows = []
    for rec in read_jsonl(path):
        if rec.get("category") not in DOLLY_CATEGORIES:
            continue
        if clean(rec.get("context", "")):
            continue
        q = clean(rec.get("instruction", ""))
        if not (40 <= len(q) <= MAX_CHARS):
            continue
        rows.append(
            {
                "query": q,
                "label": "moderate",
                "source": f"dolly_{rec['category']}",
                "answer": clean(rec.get("response", ""))[:500],
                "answer_type": "freeform",
            }
        )
    return rows


def load_gsm8k(path: Path) -> list[dict]:
    """Grade-school math word problems. The gold answer is the number after '####',
    which gives eval an exact-match check with no LLM judge needed."""
    rows = []
    for rec in read_jsonl(path):
        q = clean(rec.get("question", ""))
        raw = rec.get("answer", "")
        if "####" not in raw or not (MIN_CHARS <= len(q) <= MAX_CHARS):
            continue
        gold = clean(raw.split("####")[-1]).replace(",", "")
        rows.append(
            {
                "query": q,
                "label": "complex",
                "source": "gsm8k",
                "answer": gold,
                "answer_type": "numeric",
            }
        )
    return rows


def load_mbpp(path: Path) -> list[dict]:
    """Python programming prompts."""
    rows = []
    for rec in read_jsonl(path):
        q = clean(rec.get("text", ""))
        if not (MIN_CHARS <= len(q) <= MAX_CHARS):
            continue
        rows.append(
            {
                "query": q,
                "label": "complex",
                "source": "mbpp",
                "answer": clean(rec.get("code", ""))[:500],
                "answer_type": "code",
            }
        )
    return rows


def balanced_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n rows, spread as evenly as possible across the sources present.

    Without this, 'complex' would be ~90% GSM8K simply because GSM8K is the bigger
    file, and the model would really be learning 'is this a math problem'.
    """
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    for bucket in by_source.values():
        rng.shuffle(bucket)

    picked: list[dict] = []
    sources = sorted(by_source)
    i = 0
    while len(picked) < n and any(by_source[s] for s in sources):
        bucket = by_source[sources[i % len(sources)]]
        if bucket:
            picked.append(bucket.pop())
        i += 1
    return picked[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=PER_CLASS)
    parser.add_argument("--holdout", type=int, default=HOLDOUT_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching sources:")
    paths = {name: download(name, url) for name, url in SOURCES.items()}

    pools = {
        "simple": load_squad(paths["squad"]),
        "moderate": load_dolly(paths["dolly"]),
        "complex": load_gsm8k(paths["gsm8k"]) + load_mbpp(paths["mbpp"]),
    }

    print("\nAvailable after filtering:")
    for label, pool in pools.items():
        print(f"  {label:9s} {len(pool):6d}")
        if len(pool) < args.per_class:
            raise SystemExit(f"not enough '{label}' rows: {len(pool)} < {args.per_class}")

    rows: list[dict] = []
    for pool in pools.values():
        rows.extend(balanced_sample(pool, args.per_class, rng))

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="query").sample(frac=1.0, random_state=args.seed)
    df = df.reset_index(drop=True)

    # Stratified holdout, carved out BEFORE any training ever sees the data.
    per_class_holdout = args.holdout // df["label"].nunique()
    holdout = df.groupby("label", group_keys=False).head(per_class_holdout).reset_index(drop=True)
    train = df.drop(df.index[df["query"].isin(set(holdout["query"]))]).reset_index(drop=True)

    train.to_csv(DATA_DIR / "queries.csv", index=False)
    holdout.to_csv(DATA_DIR / "holdout.csv", index=False)

    print(f"\nqueries.csv  {len(train):4d} rows  {dict(train['label'].value_counts())}")
    print(f"holdout.csv  {len(holdout):4d} rows  {dict(holdout['label'].value_counts())}")
    print(f"sources      {dict(df['source'].value_counts())}")


if __name__ == "__main__":
    main()
