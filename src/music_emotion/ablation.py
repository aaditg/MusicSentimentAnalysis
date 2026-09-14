from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .evaluate import cv_classification
from .fusion_baseline import (
    PROCESSED,
    RAW,
    SEED,
    _audio_pipeline,
    _lyrics_vectorizer,
    load_bimodal,
)

REPORT_PATH = Path("results/ABLATION.md")


def _fused_pipeline(audio_columns) -> Pipeline:
    return Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        ("lyrics", _lyrics_vectorizer(), "text"),
                        ("audio", _audio_pipeline(), audio_columns),
                    ]
                ),
            ),
            ("model", LinearSVC(C=1.0)),
        ]
    )


def _modality_pipeline(kind: str, audio_columns) -> Pipeline:
    if kind == "lyrics":
        transformer = ("lyrics", _lyrics_vectorizer(), "text")
    else:
        transformer = ("audio", _audio_pipeline(), audio_columns)
    return Pipeline(
        [("pre", ColumnTransformer([transformer])), ("model", LinearSVC(C=1.0))]
    )


def run_ablation(raw_dir: Path = RAW, processed_dir: Path = PROCESSED) -> str:
    frame, labels, audio_columns = load_bimodal(raw_dir, processed_dir)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    scores = {}
    for name, kind in [("Lyrics only", "lyrics"), ("Audio only", "audio")]:
        result = cv_classification(
            _modality_pipeline(kind, audio_columns), frame, labels, splitter
        )
        scores[name] = result
        print(f"[ablation] {name}: {result.summary()}")

    fused = cv_classification(_fused_pipeline(audio_columns), frame, labels, splitter)
    scores["Fused"] = fused
    print(f"[ablation] Fused: {fused.summary()}")

    lyrics_f1 = scores["Lyrics only"].macro_f1_mean
    audio_f1 = scores["Audio only"].macro_f1_mean
    fused_f1 = fused.macro_f1_mean

    lines = [
        "# Modality Ablation & Emotion Pointers",
        "",
        "Bimodal MERGE set, 5-fold StratifiedKFold, LinearSVC head held fixed.",
        "",
        "## Ablation",
        "",
        "| Setting | Macro-F1 | Balanced acc | vs fused |",
        "|---|---|---|---|",
        f"| Fused (audio + lyrics) | {fused_f1:.3f} | {fused.balanced_acc_mean:.3f} | - |",
        f"| Drop audio (lyrics only) | {lyrics_f1:.3f} | "
        f"{scores['Lyrics only'].balanced_acc_mean:.3f} | {lyrics_f1 - fused_f1:+.3f} |",
        f"| Drop lyrics (audio only) | {audio_f1:.3f} | "
        f"{scores['Audio only'].balanced_acc_mean:.3f} | {audio_f1 - fused_f1:+.3f} |",
        "",
        f"- Marginal value of **adding audio** to lyrics: {fused_f1 - lyrics_f1:+.3f} macro-F1",
        f"- Marginal value of **adding lyrics** to audio: {fused_f1 - audio_f1:+.3f} macro-F1",
        "",
        "Per-class F1 (fused): "
        + ", ".join(f"{k}={v:.3f}" for k, v in fused.per_class_f1.items()),
        "",
    ]

    lines.extend(pointers_section(frame, labels, audio_columns))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {REPORT_PATH}")
    return "\n".join(lines)


def pointers_section(frame, labels, audio_columns, top_k: int = 10) -> list[str]:
    model = _fused_pipeline(audio_columns)
    model.fit(frame, labels)

    names = model.named_steps["pre"].get_feature_names_out()
    coef = model.named_steps["model"].coef_
    classes = model.named_steps["model"].classes_

    is_word = np.array([n.startswith("lyrics__word__") for n in names])
    is_audio = np.array([n.startswith("audio__") for n in names])

    def clean(name: str) -> str:
        return name.split("__")[-1]

    lines = ["## Emotion pointers (fused LinearSVC coefficients)", ""]
    for index, quadrant in enumerate(classes):
        weights = coef[index]
        top_words = _top(weights, names, is_word, top_k, clean)
        top_audio = _top(weights, names, is_audio, top_k, clean)
        lines.append(f"### {quadrant}")
        lines.append(f"- **Top lyric terms:** {', '.join(top_words)}")
        lines.append(f"- **Top audio features:** {', '.join(top_audio)}")
        lines.append("")
    return lines


def _top(weights, names, mask, k, clean):
    masked = np.where(mask, weights, -np.inf)
    order = np.argsort(masked)[::-1][:k]
    return [clean(names[i]) for i in order if np.isfinite(masked[i])]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    args = parser.parse_args()
    run_ablation(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
