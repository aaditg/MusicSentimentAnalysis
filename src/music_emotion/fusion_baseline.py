from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .evaluate import cv_classification, nested_cv_search
from .results_log import Experiment, log_experiment

PROCESSED = Path("data/processed")
RAW = Path("data/raw")
SEED = 42


class LateFusion(BaseEstimator, ClassifierMixin):

    def __init__(self, audio_columns, weight: float = 0.5, C: float = 1.0):
        self.audio_columns = audio_columns
        self.weight = weight
        self.C = C

    def _build(self):
        lyrics = Pipeline(
            [("pre", ColumnTransformer([("lyrics", _lyrics_vectorizer(), "text")])),
             ("model", LinearSVC(C=self.C))]
        )
        audio = Pipeline(
            [("pre", ColumnTransformer([("audio", _audio_pipeline(), self.audio_columns)])),
             ("model", LinearSVC(C=self.C))]
        )
        return lyrics, audio

    def fit(self, X, y):
        self.classes_ = np.array(sorted(pd.Series(y).unique()))
        self.lyrics_, self.audio_ = self._build()
        self.lyrics_.fit(X, y)
        self.audio_.fit(X, y)
        ls = self.lyrics_.decision_function(X)
        as_ = self.audio_.decision_function(X)
        self._l_mu, self._l_sd = ls.mean(0), ls.std(0) + 1e-9
        self._a_mu, self._a_sd = as_.mean(0), as_.std(0) + 1e-9
        return self

    def decision_function(self, X):
        ls = (self.lyrics_.decision_function(X) - self._l_mu) / self._l_sd
        as_ = (self.audio_.decision_function(X) - self._a_mu) / self._a_sd
        return self.weight * ls + (1.0 - self.weight) * as_

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


def _lyrics_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), max_features=20000
                ),
            ),
        ]
    )


def _audio_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def load_bimodal(raw_dir: Path, processed_dir: Path):
    labels = pd.read_csv(processed_dir / "merge_bimodal.csv")
    audio = pd.read_csv(processed_dir / "merge_bimodal_audio_features.csv")
    merged = labels.merge(audio, on="sample_id", how="inner")
    audio_columns = [c for c in audio.columns if c != "sample_id"]

    texts = [
        (raw_dir / path).read_text(encoding="utf-8", errors="replace")
        for path in merged["lyrics_path"]
    ]
    frame = merged[audio_columns].copy()
    frame["text"] = texts
    return frame, merged["quadrant"], audio_columns


def train_fusion_baseline(raw_dir: Path = RAW, processed_dir: Path = PROCESSED) -> str:
    frame, labels, audio_columns = load_bimodal(raw_dir, processed_dir)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    lyrics_only = Pipeline(
        [
            ("pre", ColumnTransformer([("lyrics", _lyrics_vectorizer(), "text")])),
            ("model", LinearSVC(C=1.0)),
        ]
    )
    audio_only = Pipeline(
        [
            ("pre", ColumnTransformer([("audio", _audio_pipeline(), audio_columns)])),
            ("model", LinearSVC(C=1.0)),
        ]
    )
    fused = Pipeline(
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

    settings = [
        ("Lyrics only", lyrics_only, "TF-IDF word+char -> LinearSVC"),
        ("Audio only", audio_only, f"{len(audio_columns)} librosa feats, scaled -> LinearSVC"),
        ("Fused (audio + lyrics)", fused, "early fusion: TF-IDF + scaled audio -> LinearSVC"),
    ]

    results = {}
    lines = [
        "MERGE Bimodal Fusion",
        "====================",
        f"Paired samples: {len(frame)}",
        f"Audio features: {len(audio_columns)}",
        "",
    ]
    for name, model, config in settings:
        result = cv_classification(model, frame, labels, splitter)
        results[name] = result
        print(f"[fusion] {name}: {result.summary()}")
        lines.append(f"{name:<24} {result.summary()}")
        log_experiment(
            Experiment(
                modality="fusion",
                name=name,
                config=f"5-fold StratifiedKFold; {config}",
                metrics=result.as_dict(),
            )
        )

    svd_fused = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        (
                            "lyrics",
                            Pipeline(
                                [
                                    ("tfidf", _lyrics_vectorizer()),
                                    ("svd", TruncatedSVD(n_components=300, random_state=SEED)),
                                ]
                            ),
                            "text",
                        ),
                        ("audio", _audio_pipeline(), audio_columns),
                    ]
                ),
            ),
            ("model", LinearSVC(C=1.0)),
        ]
    )
    result = cv_classification(svd_fused, frame, labels, splitter)
    results["Fused + SVD(300)"] = result
    print(f"[fusion] Fused + SVD(300): {result.summary()}")
    lines.append(f"{'Fused + SVD(300)':<24} {result.summary()}")
    log_experiment(
        Experiment(
            modality="fusion",
            name="Fused + SVD(300)",
            config="5-fold; TF-IDF->SVD(300) + scaled audio -> LinearSVC",
            metrics=result.as_dict(),
        )
    )

    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    late = LateFusion(audio_columns)
    grid = {"weight": [0.3, 0.5, 0.6, 0.7, 0.8], "C": [0.5, 1.0, 2.0]}
    late_result, chosen = nested_cv_search(late, grid, frame, labels, splitter, inner)
    results["Weighted late fusion"] = late_result
    print(f"[fusion] Weighted late fusion (nested CV): {late_result.summary()}")
    lines.append(f"{'Late fusion':<24} {late_result.summary()}")
    from collections import Counter

    weight_pick = Counter(str(c["weight"]) for c in chosen).most_common(1)[0][0]
    log_experiment(
        Experiment(
            modality="fusion",
            name="Weighted late fusion (nested CV)",
            config="nested 5x3; per-modality LinearSVC, z-normed scores",
            metrics=late_result.as_dict(),
            notes=f"best per-fold lyrics weight mode={weight_pick}",
        )
    )

    best_single = max(
        results["Lyrics only"].macro_f1_mean, results["Audio only"].macro_f1_mean
    )
    delta = results["Fused (audio + lyrics)"].macro_f1_mean - best_single
    lines.append("")
    lines.append(
        f"Fusion vs best single modality (macro-F1): {delta:+.3f}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    args = parser.parse_args()
    print(train_fusion_baseline(args.raw_dir, args.processed_dir))


if __name__ == "__main__":
    main()
