from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def read_lyrics(raw_dir: Path, frame: pd.DataFrame) -> list[str]:
    return [
        (raw_dir / path).read_text(encoding="utf-8", errors="replace")
        for path in frame["lyrics_path"]
    ]


def read_split(root: Path, name: str) -> set[str]:
    path = root / "tvt_dataframes" / "tvt_70_15_15" / f"tvt_70_15_15_{name}_lyrics_complete.csv"
    return set(pd.read_csv(path)["Song"])


def train_lyrics_baseline(data_path: Path, raw_dir: Path) -> str:
    frame = pd.read_csv(data_path)
    root = raw_dir / "merge" / "MERGE_Lyrics_Complete"
    train_ids = read_split(root, "train") | read_split(root, "validate")
    test_ids = read_split(root, "test")
    train = frame[frame["sample_id"].isin(train_ids)]
    test = frame[frame["sample_id"].isin(test_ids)]

    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            (
                "char",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000),
            ),
        ]
    )
    model = Pipeline(
        [
            ("tfidf", features),
            ("model", LinearSVC(C=1.0)),
        ]
    )
    model.fit(read_lyrics(raw_dir, train), train["quadrant"])
    predictions = model.predict(read_lyrics(raw_dir, test))

    lines = [
        "MERGE Lyrics Baseline",
        "=====================",
        f"Train lyrics: {len(train)}",
        f"Test lyrics: {len(test)}",
        f"Accuracy: {accuracy_score(test['quadrant'], predictions):.3f}",
        "",
        classification_report(test["quadrant"], predictions, zero_division=0),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/merge_lyrics.csv"),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    print(train_lyrics_baseline(args.data_path, args.raw_dir))


if __name__ == "__main__":
    main()
