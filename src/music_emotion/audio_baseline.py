from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split

from .labels import quadrant_from_values

LABEL_COLUMNS = {
    "sample_id",
    "source",
    "quadrant",
    "valence_class",
    "arousal_class",
    "valence_mean",
    "valence_std",
    "arousal_mean",
    "arousal_std",
}

_NON_FEATURE_EXACT = {"sample_id", "source"}
_NON_FEATURE_TOKENS = ("valence", "arousal", "quadrant")


def audio_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for name in frame.columns:
        lowered = name.lower()
        if name in _NON_FEATURE_EXACT:
            continue
        if any(token in lowered for token in _NON_FEATURE_TOKENS):
            continue
        columns.append(name)
    return columns


def regression_line(name: str, actual: pd.Series, predicted: np.ndarray) -> str:
    correlation = np.corrcoef(actual, predicted)[0, 1]
    mae = mean_absolute_error(actual, predicted)
    return f"{name:<8} MAE={mae:.3f} correlation={correlation:.3f}"


def train_audio_baseline(data_path: Path) -> str:
    frame = pd.read_csv(data_path)
    feature_columns = audio_feature_columns(frame)
    features = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median())
    train, test = train_test_split(
        frame.index,
        test_size=0.2,
        random_state=42,
        stratify=frame["quadrant"],
    )

    valence = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    arousal = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    valence.fit(features.loc[train], frame.loc[train, "valence_mean"])
    arousal.fit(features.loc[train], frame.loc[train, "arousal_mean"])

    valence_pred = valence.predict(features.loc[test])
    arousal_pred = arousal.predict(features.loc[test])
    predictions = [
        quadrant_from_values(v, a, midpoint=5.0) for v, a in zip(valence_pred, arousal_pred)
    ]

    importance = pd.Series(
        (valence.feature_importances_ + arousal.feature_importances_) / 2,
        index=feature_columns,
    )
    top_features = importance.nlargest(10)
    lines = [
        "DEAM Audio Baseline",
        "===================",
        f"Train songs: {len(train)}",
        f"Test songs: {len(test)}",
        f"Audio features: {len(feature_columns)}",
        "Model: valence/arousal regression -> quadrant (threshold at 5.0)",
        f"Quadrant accuracy: {accuracy_score(frame.loc[test, 'quadrant'], predictions):.3f}",
        "Balanced accuracy: "
        f"{balanced_accuracy_score(frame.loc[test, 'quadrant'], predictions):.3f}",
        "",
        classification_report(frame.loc[test, "quadrant"], predictions, zero_division=0),
        "Regression",
        "----------",
        regression_line("Valence", frame.loc[test, "valence_mean"], valence_pred),
        regression_line("Arousal", frame.loc[test, "arousal_mean"], arousal_pred),
        "",
        "Top audio features (mean of valence + arousal importance)",
        "---------------------------------------------------------",
    ]
    lines.extend(f"{name}: {score:.4f}" for name, score in top_features.items())
    return "\n".join(lines)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/deam_audio_features.csv"),
    )
    args = parser.parse_args()
    print(train_audio_baseline(args.data_path))


if __name__ == "__main__":
    main()
