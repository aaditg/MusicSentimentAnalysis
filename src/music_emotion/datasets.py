from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetCard:
    name: str
    modality: str
    labels: str
    use_case: str
    note: str


DATASETS = [
    DatasetCard(
        name="DEAM",
        modality="audio",
        labels="valence, arousal",
        use_case="audio benchmark",
        note="Best first set for audio emotion regression.",
    ),
    DatasetCard(
        name="MoodyLyrics",
        modality="lyrics",
        labels="quadrant",
        use_case="lyrics baseline",
        note="Good first text-only benchmark.",
    ),
    DatasetCard(
        name="MERGE",
        modality="audio + lyrics",
        labels="quadrant, valence, arousal",
        use_case="multimodal benchmark",
        note="Best direct fit for comparing single vs combined inputs.",
    ),
    DatasetCard(
        name="EMOPIA",
        modality="symbolic + audio",
        labels="quadrant",
        use_case="symbolic music benchmark",
        note="Strong choice for chord and MIDI-based analysis.",
    ),
]


def format_dataset_table() -> str:
    lines = []
    lines.append("Datasets")
    lines.append("-" * 84)
    lines.append(f"{'Name':<14}{'Modality':<20}{'Labels':<30}{'Use case'}")
    lines.append("-" * 84)
    for item in DATASETS:
        lines.append(
            f"{item.name:<14}{item.modality:<20}{item.labels:<30}{item.use_case}"
        )
    return "\n".join(lines)
