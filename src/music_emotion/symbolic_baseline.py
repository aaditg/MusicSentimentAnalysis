from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_symbolic_baseline(data_path: Path) -> str:
    frame = pd.read_csv(data_path)
    category_columns = ["key", "mode"]
    ignored = {
        "sample_id",
        "source",
        "song_group",
        "quadrant",
        "valence_class",
        "arousal_class",
    }
    number_columns = [
        name for name in frame if name not in ignored and name not in category_columns
    ]
    features = frame[category_columns + number_columns]
    labels = frame["quadrant"]
    groups = frame["song_group"]

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(split.split(features, labels, groups))

    processor = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore"), ["key", "mode"]),
            ("number", StandardScaler(), number_columns),
        ]
    )
    model = Pipeline(
        [
            ("processor", processor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(features.iloc[train_index], labels.iloc[train_index])
    predictions = model.predict(features.iloc[test_index])

    lines = [
        "EMOPIA Symbolic Baseline",
        "========================",
        f"Train clips: {len(train_index)}",
        f"Test clips: {len(test_index)}",
        f"Symbolic features: {len(features.columns)}",
        f"Accuracy: {accuracy_score(labels.iloc[test_index], predictions):.3f}",
        "",
        classification_report(labels.iloc[test_index], predictions, zero_division=0),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/emopia.csv"),
    )
    args = parser.parse_args()
    print(train_symbolic_baseline(args.data_path))


if __name__ == "__main__":
    main()
