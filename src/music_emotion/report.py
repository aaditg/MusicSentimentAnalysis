from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from .inventory import format_inventory


def format_processed(processed_dir: Path) -> str:
    lines = ["", "Processed Tables", "================"]
    for path in sorted(processed_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        lines.append(f"{path.name}: {len(frame)} rows")
        if "quadrant" in frame:
            counts = frame["quadrant"].value_counts().sort_index()
            lines.append("  " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    return "\n".join(lines)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    print(format_inventory(args.raw_dir))
    print(format_processed(args.processed_dir))


if __name__ == "__main__":
    main()
