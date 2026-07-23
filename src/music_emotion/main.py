from .datasets import DATASETS, format_dataset_table
from .roadmap import format_roadmap


def format_dataset_notes() -> str:
    lines = ["", "Dataset Notes", "-" * 13]
    for item in DATASETS:
        lines.append(f"{item.name}: {item.note}")
    return "\n".join(lines)


def main() -> None:
    parts = [
        "Music Emotion Research Starter",
        "=============================",
        "",
        "Model tracks",
        "------------",
        "1. symbolic music",
        "2. audio",
        "3. lyrics",
        "4. multimodal fusion",
        "",
        format_roadmap(),
        "",
        format_dataset_table(),
        format_dataset_notes(),
    ]
    print("\n".join(parts))


if __name__ == "__main__":
    main()
