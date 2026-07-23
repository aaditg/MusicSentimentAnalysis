from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def aggregate_file(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, sep=";")
    values = frame.drop(columns=["frameTime"], errors="ignore")
    row = values.mean(numeric_only=True).to_dict()
    row["sample_id"] = int(path.stem)
    return row


def aggregate_deam(raw_dir: Path, processed_dir: Path) -> Path:
    features_dir = raw_dir / "deam" / "features"
    paths = sorted(features_dir.glob("*.csv"), key=lambda path: int(path.stem))
    if not paths:
        raise FileNotFoundError("DEAM openSMILE features are missing")

    rows = []
    for index, path in enumerate(paths, start=1):
        rows.append(aggregate_file(path))
        if index % 100 == 0 or index == len(paths):
            print(f"aggregate audio: {index}/{len(paths)}")

    labels = pd.read_csv(processed_dir / "deam_labels.csv")
    frame = pd.DataFrame(rows).merge(labels, on="sample_id")
    output = processed_dir / "deam_audio_features.csv"
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    return output


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    aggregate_deam(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
