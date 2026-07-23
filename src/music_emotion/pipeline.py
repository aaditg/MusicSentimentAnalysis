from pathlib import Path

from .audio_baseline import train_audio_baseline
from .audio_features import aggregate_deam
from .lyrics_baseline import train_lyrics_baseline
from .midi_features import aggregate_emopia
from .normalize import normalize_all
from .report import format_processed
from .symbolic_baseline import train_symbolic_baseline


def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    normalize_all(raw_dir, processed_dir)

    audio_path = processed_dir / "deam_audio_features.csv"
    if not audio_path.exists():
        aggregate_deam(raw_dir, processed_dir)

    midi_path = processed_dir / "emopia_midi_features.csv"
    if not midi_path.exists():
        aggregate_emopia(raw_dir, processed_dir)

    sections = [
        format_processed(processed_dir),
        "",
        train_symbolic_baseline(midi_path),
        "",
        train_lyrics_baseline(processed_dir / "merge_lyrics.csv", raw_dir),
        "",
        train_audio_baseline(audio_path),
    ]
    print("\n".join(sections))


if __name__ == "__main__":
    main()
