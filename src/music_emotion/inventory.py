from argparse import ArgumentParser
from pathlib import Path


def file_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


def file_size(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def format_inventory(raw_dir: Path) -> str:
    lines = ["Raw Data Inventory", "=================="]
    for path in sorted(item for item in raw_dir.iterdir() if item.is_dir()):
        if path.name == "_archives":
            continue
        size_mb = file_size(path) / 1024 / 1024
        lines.append(f"{path.name:<12} {file_count(path):>7} files {size_mb:>10.1f} MB")
    return "\n".join(lines)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    print(format_inventory(args.raw_dir))


if __name__ == "__main__":
    main()
