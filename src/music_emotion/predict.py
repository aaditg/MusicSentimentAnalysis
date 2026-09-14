"""Train once, then score any song from an audio file plus its lyrics."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .fusion_baseline import AudioOnly, LateFusion, load_bimodal
from .labels import QUADRANTS
from .merge_audio_features import clip_features

PROCESSED = Path("data/processed")
RAW = Path("data/raw")
MODEL_PATH = Path("models/fusion.joblib")

# Chosen by nested CV in fusion_baseline (lyrics weight mode 0.6).
LYRICS_WEIGHT = 0.6
C = 1.0


def train(
    raw_dir: Path = RAW,
    processed_dir: Path = PROCESSED,
    model_path: Path = MODEL_PATH,
    audio_only: bool = False,
) -> Path:
    frame, labels, audio_columns = load_bimodal(raw_dir, processed_dir, with_lyrics=not audio_only)
    if audio_only:
        model = AudioOnly(audio_columns, C=C)
    else:
        model = LateFusion(audio_columns, weight=LYRICS_WEIGHT, C=C)
    model.fit(frame, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "audio_columns": audio_columns, "audio_only": audio_only},
        model_path,
    )
    kind = "audio-only" if audio_only else "audio + lyrics"
    print(f"saved: {model_path} ({kind}, trained on {len(frame)} songs)")
    return model_path


def _row(audio_path: Path, lyrics_path: Path, audio_columns: list[str]) -> pd.DataFrame:
    features = clip_features(audio_path)
    frame = pd.DataFrame([features]).reindex(columns=audio_columns)
    frame["text"] = lyrics_path.read_text(encoding="utf-8", errors="replace")
    return frame


def predict(audio_path: Path, lyrics_path: Path, model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} is missing - run `python -m music_emotion.predict train` first"
        )
    bundle = joblib.load(model_path)
    model, audio_columns = bundle["model"], bundle["audio_columns"]

    frame = _row(audio_path, lyrics_path, audio_columns)
    scores = model.decision_function(frame)[0]
    quadrant = str(model.classes_[int(np.argmax(scores))])
    valence, arousal = QUADRANTS[quadrant]

    lyrics_scores = model.lyrics_.decision_function(frame)[0]
    audio_scores = model.audio_.decision_function(frame)[0]
    per_class = {str(c): float(s) for c, s in zip(model.classes_, scores)}
    return {
        "quadrant": quadrant,
        "valence": valence,
        "arousal": arousal,
        "scores": per_class,
        "lyrics_only": str(model.classes_[int(np.argmax(lyrics_scores))]),
        "audio_only": str(model.classes_[int(np.argmax(audio_scores))]),
    }


def format_prediction(result: dict) -> str:
    lines = [
        f"Quadrant:  {result['quadrant']}  "
        f"(valence {result['valence']}, arousal {result['arousal']})",
        "",
        "Fused scores:",
    ]
    for name, score in sorted(result["scores"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name}  {score:+.3f}")
    lines += [
        "",
        f"Lyrics alone: {result['lyrics_only']}",
        f"Audio alone:  {result['audio_only']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="fit the fusion model on MERGE and save it")
    train_parser.add_argument("--raw-dir", type=Path, default=RAW)
    train_parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    train_parser.add_argument("--model", type=Path, default=MODEL_PATH)
    train_parser.add_argument(
        "--audio-only",
        action="store_true",
        help="fit only the audio branch; skips reading the lyrics corpus",
    )

    run_parser = sub.add_parser("song", help="score one song")
    run_parser.add_argument("--audio", type=Path, required=True)
    run_parser.add_argument("--lyrics", type=Path, required=True)
    run_parser.add_argument("--model", type=Path, default=MODEL_PATH)

    args = parser.parse_args()
    if args.command == "train":
        train(args.raw_dir, args.processed_dir, args.model, args.audio_only)
    else:
        print(format_prediction(predict(args.audio, args.lyrics, args.model)))


if __name__ == "__main__":
    main()
