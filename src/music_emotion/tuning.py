from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from .audio_baseline import audio_feature_columns
from .evaluate import CVResult, cv_classification, nested_cv_search
from .labels import quadrant_from_values
from .results_log import Experiment, log_experiment

PROCESSED = Path("data/processed")
RAW = Path("data/raw")
SEED = 42


def _summarize_params(chosen: list[dict]) -> str:
    from collections import Counter

    if not chosen:
        return ""
    parts = []
    for key in chosen[0]:
        counts = Counter(str(fold[key]) for fold in chosen)
        value = counts.most_common(1)[0][0]
        parts.append(f"{key.split('__')[-1]}={value}")
    return "best per-fold: " + ", ".join(parts)


def load_symbolic():
    frame = pd.read_csv(PROCESSED / "emopia_midi_features.csv")
    category = ["key", "mode"]
    ignored = {
        "sample_id",
        "source",
        "song_group",
        "quadrant",
        "valence_class",
        "arousal_class",
    }
    numbers = [c for c in frame if c not in ignored and c not in category]
    return frame, category, numbers


def symbolic_pipeline(model, numbers):
    pre = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore"), ["key", "mode"]),
            ("number", StandardScaler(), numbers),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def run_symbolic():
    frame, category, numbers = load_symbolic()
    X = frame[category + numbers]
    y = frame["quadrant"]
    groups = frame["song_group"]
    splitter = GroupKFold(n_splits=5)

    trials = [
        ("LogReg (baseline)", LogisticRegression(max_iter=1000)),
        (
            "LogReg balanced",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        (
            "LogReg balanced C=2",
            LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=-1
            ),
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(random_state=SEED),
        ),
    ]

    n_feats = len(category) + len(numbers)
    for name, model in trials:
        result = cv_classification(
            symbolic_pipeline(model, numbers), X, y, splitter, groups=groups
        )
        print(f"[symbolic] {name}: {result.summary()}")
        log_experiment(
            Experiment(
                modality="symbolic",
                name=name,
                config=f"5-fold GroupKFold by song_group; {n_feats} feats",
                metrics=result.as_dict(),
            )
        )

    inner = GroupKFold(n_splits=3)
    sweeps = [
        (
            "LogReg tuned (nested CV)",
            symbolic_pipeline(LogisticRegression(max_iter=3000), numbers),
            {
                "model__C": [0.25, 0.5, 1.0, 2.0, 4.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        (
            "HistGB tuned (nested CV)",
            symbolic_pipeline(HistGradientBoostingClassifier(random_state=SEED), numbers),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_iter": [200, 400],
                "model__l2_regularization": [0.0, 1.0],
            },
        ),
    ]
    for name, estimator, grid in sweeps:
        result, chosen = nested_cv_search(
            estimator, grid, X, y, splitter, inner, groups=groups
        )
        print(f"[symbolic] {name}: {result.summary()}")
        log_experiment(
            Experiment(
                modality="symbolic",
                name=name,
                config=f"nested 5x3 GroupKFold; expanded {n_feats} feats",
                metrics=result.as_dict(),
                notes=_summarize_params(chosen),
            )
        )


def load_lyrics():
    frame = pd.read_csv(PROCESSED / "merge_lyrics.csv")
    texts = [
        (RAW / path).read_text(encoding="utf-8", errors="replace")
        for path in frame["lyrics_path"]
    ]
    return texts, frame["quadrant"]


def run_lyrics():
    texts, y = load_lyrics()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    word = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    word_char = FeatureUnion(
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

    trials = [
        (
            "LogReg word 1-2gram (baseline)",
            Pipeline([("tfidf", word), ("model", LogisticRegression(max_iter=1000))]),
        ),
        (
            "LogReg balanced",
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                    (
                        "model",
                        LogisticRegression(max_iter=1000, class_weight="balanced", C=3.0),
                    ),
                ]
            ),
        ),
        (
            "LinearSVC word+char",
            Pipeline([("tfidf", word_char), ("model", LinearSVC(C=1.0))]),
        ),
        (
            "ComplementNB word",
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
                    ("model", ComplementNB()),
                ]
            ),
        ),
    ]

    for name, model in trials:
        result = cv_classification(model, texts, y, splitter)
        print(f"[lyrics] {name}: {result.summary()}")
        log_experiment(
            Experiment(
                modality="lyrics",
                name=name,
                config="5-fold StratifiedKFold; TF-IDF features",
                metrics=result.as_dict(),
            )
        )

    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    estimator = Pipeline(
        [
            (
                "tfidf",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb", ngram_range=(3, 5), max_features=20000
                            ),
                        ),
                    ]
                ),
            ),
            ("model", LinearSVC()),
        ]
    )
    grid = {
        "model__C": [0.5, 1.0, 2.0],
        "model__class_weight": [None, "balanced"],
        "tfidf__word__sublinear_tf": [False, True],
        "tfidf__word__ngram_range": [(1, 2), (1, 3)],
    }
    result, chosen = nested_cv_search(estimator, grid, texts, y, splitter, inner)
    print(f"[lyrics] LinearSVC tuned (nested CV): {result.summary()}")
    log_experiment(
        Experiment(
            modality="lyrics",
            name="LinearSVC tuned (nested CV)",
            config="nested 5x3 StratifiedKFold; TF-IDF word+char",
            metrics=result.as_dict(),
            notes=_summarize_params(chosen),
        )
    )


def load_audio():
    frame = pd.read_csv(PROCESSED / "deam_audio_features.csv")
    feature_cols = audio_feature_columns(frame)
    X = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    return frame, X, feature_cols


def run_audio():
    frame, X, feature_cols = load_audio()
    y = frame["quadrant"]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    impute = ("impute", SimpleImputer(strategy="median"))

    trials = [
        (
            "RandomForest balanced (baseline)",
            Pipeline(
                [
                    impute,
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=250,
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "RandomForest balanced_subsample",
            Pipeline(
                [
                    impute,
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            class_weight="balanced_subsample",
                            min_samples_leaf=2,
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "HistGradientBoosting",
            Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_iter=400,
                            l2_regularization=1.0,
                            random_state=SEED,
                        ),
                    )
                ]
            ),
        ),
        (
            "LogReg balanced (scaled, top-80)",
            Pipeline(
                [
                    impute,
                    ("scale", StandardScaler()),
                    ("select", SelectKBest(f_classif, k=80)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000, class_weight="balanced", C=1.0
                        ),
                    ),
                ]
            ),
        ),
        (
            "RandomForest top-80 features",
            Pipeline(
                [
                    impute,
                    ("scale", MinMaxScaler()),
                    ("select", SelectKBest(f_classif, k=80)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            class_weight="balanced_subsample",
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]

    trials.append(
        (
            "RandomForest + SMOTE",
            ImbPipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500, random_state=SEED, n_jobs=-1
                        ),
                    ),
                ]
            ),
        )
    )
    trials.append(
        (
            "HistGB + SMOTE (scaled)",
            ImbPipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05, max_iter=400, random_state=SEED
                        ),
                    ),
                ]
            ),
        )
    )

    for name, model in trials:
        result = cv_classification(model, X, y, splitter)
        print(f"[audio] {name}: {result.summary()}")
        log_experiment(
            Experiment(
                modality="audio",
                name=name,
                config=f"5-fold StratifiedKFold; {len(feature_cols)} openSMILE feats",
                metrics=result.as_dict(),
                primary_metric="balanced_acc_mean",
            )
        )

    run_audio_regression_quadrant(frame, X, feature_cols, splitter)
    run_audio_regression_tuned(frame, X, feature_cols, splitter)


def run_audio_regression_quadrant(frame, X, feature_cols, splitter):
    y = frame["quadrant"].reset_index(drop=True)
    valence = frame["valence_mean"].reset_index(drop=True)
    arousal = frame["arousal_mean"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    labels = sorted(y.unique())

    accuracies, macro_f1s, balanced = [], [], []
    per_class = {label: [] for label in labels}

    for train_index, test_index in splitter.split(np.zeros(len(y)), y):
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(X.iloc[train_index])
        x_test = imputer.transform(X.iloc[test_index])

        v_model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        a_model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        v_model.fit(x_train, valence.iloc[train_index])
        a_model.fit(x_train, arousal.iloc[train_index])

        v_pred = v_model.predict(x_test)
        a_pred = a_model.predict(x_test)
        predicted = [
            quadrant_from_values(v, a, midpoint=5.0) for v, a in zip(v_pred, a_pred)
        ]
        actual = y.iloc[test_index]

        accuracies.append(accuracy_score(actual, predicted))
        macro_f1s.append(f1_score(actual, predicted, average="macro", zero_division=0))
        balanced.append(balanced_accuracy_score(actual, predicted))
        fold_f1 = f1_score(actual, predicted, labels=labels, average=None, zero_division=0)
        for label, score in zip(labels, fold_f1):
            per_class[label].append(score)

    result = CVResult(
        accuracy_mean=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies)),
        macro_f1_mean=float(np.mean(macro_f1s)),
        macro_f1_std=float(np.std(macro_f1s)),
        balanced_acc_mean=float(np.mean(balanced)),
        balanced_acc_std=float(np.std(balanced)),
        n_splits=splitter.get_n_splits(),
        per_class_f1={k: float(np.mean(v)) for k, v in per_class.items()},
    )
    print(f"[audio] Regression-derived quadrant: {result.summary()}")
    log_experiment(
        Experiment(
            modality="audio",
            name="Regression-derived quadrant",
            config=f"5-fold; 2x RF regressor on V/A, threshold at 5.0; {len(feature_cols)} feats",
            metrics=result.as_dict(),
            primary_metric="balanced_acc_mean",
            notes=(
                "Quadrant inferred from predicted valence/arousal "
                "rather than direct classification."
            ),
        )
    )


def _quadrants_from_thresholds(v_pred, a_pred, v_thr, a_thr):
    v_high = v_pred >= v_thr
    a_high = a_pred >= a_thr
    return np.select(
        [v_high & a_high, ~v_high & a_high, ~v_high & ~a_high],
        ["Q1", "Q2", "Q3"],
        default="Q4",
    )


def _best_thresholds(v_pred, a_pred, actual, grid):
    best_score, best = -1.0, (5.0, 5.0)
    actual = np.asarray(actual)
    for v_thr in grid:
        for a_thr in grid:
            predicted = _quadrants_from_thresholds(v_pred, a_pred, v_thr, a_thr)
            score = f1_score(actual, predicted, average="macro", zero_division=0)
            if score > best_score:
                best_score, best = score, (float(v_thr), float(a_thr))
    return best


def run_audio_regression_tuned(frame, X, feature_cols, splitter):
    y = frame["quadrant"].reset_index(drop=True)
    valence = frame["valence_mean"].reset_index(drop=True)
    arousal = frame["arousal_mean"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    labels = sorted(y.unique())
    grid = np.linspace(4.0, 6.0, 17)

    accuracies, macro_f1s, balanced = [], [], []
    per_class = {label: [] for label in labels}
    chosen = []

    for train_index, test_index in splitter.split(np.zeros(len(y)), y):
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(X.iloc[train_index])
        x_test = imputer.transform(X.iloc[test_index])

        v_model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        a_model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)

        oof_v = cross_val_predict(v_model, x_train, valence.iloc[train_index], cv=3, n_jobs=-1)
        oof_a = cross_val_predict(a_model, x_train, arousal.iloc[train_index], cv=3, n_jobs=-1)
        v_thr, a_thr = _best_thresholds(oof_v, oof_a, y.iloc[train_index], grid)
        chosen.append({"v_thr": round(v_thr, 2), "a_thr": round(a_thr, 2)})

        v_model.fit(x_train, valence.iloc[train_index])
        a_model.fit(x_train, arousal.iloc[train_index])
        predicted = _quadrants_from_thresholds(
            v_model.predict(x_test), a_model.predict(x_test), v_thr, a_thr
        )
        actual = y.iloc[test_index]

        accuracies.append(accuracy_score(actual, predicted))
        macro_f1s.append(f1_score(actual, predicted, average="macro", zero_division=0))
        balanced.append(balanced_accuracy_score(actual, predicted))
        fold_f1 = f1_score(actual, predicted, labels=labels, average=None, zero_division=0)
        for label, score in zip(labels, fold_f1):
            per_class[label].append(score)

    result = CVResult(
        accuracy_mean=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies)),
        macro_f1_mean=float(np.mean(macro_f1s)),
        macro_f1_std=float(np.std(macro_f1s)),
        balanced_acc_mean=float(np.mean(balanced)),
        balanced_acc_std=float(np.std(balanced)),
        n_splits=splitter.get_n_splits(),
        per_class_f1={k: float(np.mean(v)) for k, v in per_class.items()},
    )
    print(f"[audio] Regression-derived + tuned thresholds: {result.summary()}")
    log_experiment(
        Experiment(
            modality="audio",
            name="Regression-derived + tuned thresholds",
            config=f"5-fold; 2x RF regressor on V/A, learned thresholds; {len(feature_cols)} feats",
            metrics=result.as_dict(),
            primary_metric="balanced_acc_mean",
            notes=_summarize_params(chosen),
        )
    )


RUNNERS = {"symbolic": run_symbolic, "lyrics": run_lyrics, "audio": run_audio}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--modality", choices=["all", *RUNNERS], default="all"
    )
    args = parser.parse_args()
    targets = RUNNERS.keys() if args.modality == "all" else [args.modality]
    for name in targets:
        print(f"\n=== Tuning: {name} ===")
        RUNNERS[name]()


if __name__ == "__main__":
    main()
