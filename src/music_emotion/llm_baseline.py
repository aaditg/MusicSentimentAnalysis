from __future__ import annotations

import json
import os
import re
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from .labels import QUADRANTS
from .lyrics_transformer import RAW, PROCESSED, load_lyrics_split, metric_block
from .results_log import Experiment, log_experiment

CACHE_DIR = Path("results/_llm_cache")
MAX_CHARS = 3000
N_FEW_SHOT = 4
MAX_WORKERS = 8
MAX_RETRIES = 5
SEED = 42

QUADRANT_GUIDE = {
    "Q1": "high valence, high arousal -- happy, joyful, excited, energetic, upbeat",
    "Q2": "low valence, high arousal -- angry, tense, anxious, aggressive, agitated",
    "Q3": "low valence, low arousal -- sad, depressed, tired, melancholic, gloomy",
    "Q4": "high valence, low arousal -- calm, peaceful, relaxed, tender, content",
}

SYSTEM_PROMPT = (
    "You are an expert music-emotion annotator. Classify song lyrics into one of "
    "four emotion quadrants from Russell's valence-arousal model:\n"
    + "\n".join(f"- {q}: {desc}" for q, desc in QUADRANT_GUIDE.items())
    + "\n\nReply with ONLY the quadrant code (Q1, Q2, Q3, or Q4) and nothing else."
)

_LABEL_RE = re.compile(r"Q[1-4]")


def _truncate(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS]


def build_few_shot(train: pd.DataFrame) -> list[dict]:
    messages: list[dict] = []
    rng = train.sample(frac=1.0, random_state=SEED)
    for quadrant in QUADRANTS:
        match = rng[rng["quadrant"] == quadrant]
        if match.empty:
            continue
        row = match.iloc[0]
        snippet = _truncate(row["text"])[:800]
        messages.append({"role": "user", "content": f"Lyrics:\n{snippet}"})
        messages.append({"role": "assistant", "content": quadrant})
    return messages


def parse_quadrant(content: str) -> str | None:
    if not content:
        return None
    match = _LABEL_RE.search(content.upper())
    return match.group(0) if match else None


MAX_COMPLETION_TOKENS = 2000
_PARAM_ERR_HINTS = ("temperature", "reasoning_effort", "unsupported", "max_tokens")
_WORKING_KWARGS: dict[str, dict] = {}


def _kwarg_variants(model: str) -> list[dict]:
    cached = _WORKING_KWARGS.get(model)
    if cached is not None:
        return [cached]
    return [
        {"temperature": 0},
        {"reasoning_effort": "none"},
        {},
    ]


def classify_one(client, model: str, shots: list[dict], lyrics: str) -> str | None:
    user_msg = {"role": "user", "content": f"Lyrics:\n{_truncate(lyrics)}"}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *shots, user_msg]
    base = {"model": model, "messages": messages, "max_completion_tokens": MAX_COMPLETION_TOKENS}
    for attempt in range(MAX_RETRIES):
        param_failed = False
        for extra in _kwarg_variants(model):
            try:
                resp = client.chat.completions.create(**base, **extra)
                _WORKING_KWARGS.setdefault(model, extra)
                return parse_quadrant(resp.choices[0].message.content)
            except Exception as exc:
                if any(hint in str(exc).lower() for hint in _PARAM_ERR_HINTS):
                    param_failed = True
                    continue
                wait = min(2 ** attempt, 30)
                print(f"  retry {attempt + 1}/{MAX_RETRIES} after error: {exc} (sleep {wait}s)")
                time.sleep(wait)
                break
        else:
            if param_failed:
                return None
    return None


def _cache_path(model: str, strategy: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return CACHE_DIR / f"{safe}__{strategy}.jsonl"


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec["sample_id"]] = rec["predicted"]
    return out


def run_model(client, model: str, strategy: str, train, test) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(model, strategy)
    cache = _load_cache(cache_path)
    shots = build_few_shot(train) if strategy == "few" else []

    todo = [r for r in test.itertuples() if str(r.sample_id) not in cache]
    print(
        f"[{model} | {strategy}-shot] {len(test)} test samples, "
        f"{len(cache)} cached, {len(todo)} to query"
    )

    if todo:
        handle = cache_path.open("a", encoding="utf-8")
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {
                    pool.submit(classify_one, client, model, shots, row.text): row
                    for row in todo
                }
                done = 0
                for future in as_completed(futures):
                    row = futures[future]
                    predicted = future.result() or "Q3"
                    cache[str(row.sample_id)] = predicted
                    handle.write(
                        json.dumps({"sample_id": str(row.sample_id), "predicted": predicted})
                        + "\n"
                    )
                    handle.flush()
                    done += 1
                    if done % 25 == 0:
                        print(f"    {done}/{len(todo)} done")
        finally:
            handle.close()

    actual = test["quadrant"].tolist()
    predicted = [cache[str(sid)] for sid in test["sample_id"]]
    block = metric_block(actual, predicted)
    block["_report"] = classification_report(actual, predicted, zero_division=0)
    return block


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    parser.add_argument("--models", nargs="+", default=["gpt-5.5", "gpt-4.1-mini"])
    parser.add_argument("--shots", nargs="+", default=["zero", "few"], choices=["zero", "few"])
    parser.add_argument("--limit", type=int, default=0, help="cap test samples (0 = all)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running.")

    from openai import OpenAI

    client = OpenAI()

    train, test = load_lyrics_split(args.raw_dir, args.processed_dir)
    if args.limit:
        per = max(1, args.limit // test["quadrant"].nunique())
        keep = []
        for _, group in test.groupby("quadrant"):
            keep.extend(group.sample(min(len(group), per), random_state=SEED).index)
        test = test.loc[keep].reset_index(drop=True)
    print(f"train lyrics: {len(train)} | test lyrics: {len(test)}")

    strategy_label = {"zero": "zero-shot", "few": "few-shot"}
    for model in args.models:
        for strategy in args.shots:
            block = run_model(client, model, strategy, train, test)
            report = block.pop("_report")
            print(
                f"[{model} | {strategy}-shot] acc={block['accuracy_mean']:.3f} "
                f"macroF1={block['macro_f1_mean']:.3f}"
            )
            print(report)
            log_experiment(
                Experiment(
                    modality="llm",
                    name=f"{model} ({strategy_label[strategy]})",
                    config=(
                        f"MERGE official 70/15/15 test (n={len(test)}); "
                        f"{strategy_label[strategy]} prompt"
                    ),
                    metrics=block,
                )
            )


if __name__ == "__main__":
    main()
