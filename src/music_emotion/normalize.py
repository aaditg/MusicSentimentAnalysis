from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from .labels import label_columns, quadrant_from_values


def find_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name}, found {len(matches)}")
    return matches[0]


def normalize_emopia(raw_dir: Path, output_dir: Path) -> Path:
    source = find_one(raw_dir / "emopia", "key_mode_tempo.csv")
    frame = pd.read_csv(source)
    frame = frame.rename(
        columns={
            "name": "sample_id",
            "keyname": "key",
            "keymode": "mode",
            "label": "quadrant",
        }
    )
    frame = frame[["sample_id", "key", "mode", "tempo", "quadrant"]].copy()
    frame["source"] = "emopia"
    frame["song_group"] = frame["sample_id"].str.rsplit("_", n=1).str[0]
    frame[["valence_class", "arousal_class"]] = frame["quadrant"].apply(
        lambda value: pd.Series(label_columns(value))
    )
    output = output_dir / "emopia.csv"
    frame.to_csv(output, index=False)
    return output


def normalize_deam(raw_dir: Path, output_dir: Path) -> Path:
    source_dir = (
        raw_dir
        / "deam"
        / "annotations"
        / "annotations averaged per song"
        / "song_level"
    )
    files = sorted(source_dir.glob("static_annotations_averaged_songs_*.csv"))
    if not files:
        raise FileNotFoundError("DEAM static labels are missing")
    frame = pd.concat([pd.read_csv(path, skipinitialspace=True) for path in files])
    frame.columns = frame.columns.str.strip()
    frame = frame.rename(columns={"song_id": "sample_id"})
    frame["source"] = "deam"
    frame["quadrant"] = frame.apply(
        lambda row: quadrant_from_values(
            row["valence_mean"], row["arousal_mean"], midpoint=5.0
        ),
        axis=1,
    )
    frame[["valence_class", "arousal_class"]] = frame["quadrant"].apply(
        lambda value: pd.Series(label_columns(value))
    )
    output = output_dir / "deam_labels.csv"
    frame.to_csv(output, index=False)
    return output


def normalize_merge_lyrics(raw_dir: Path, output_dir: Path) -> Path:
    root = raw_dir / "merge" / "MERGE_Lyrics_Complete"
    metadata = pd.read_csv(root / "merge_lyrics_complete_metadata.csv")
    values = pd.read_csv(root / "merge_lyrics_complete_av_values.csv")
    frame = metadata[["Song", "Quadrant", "Artist", "Title"]].merge(values, on="Song")
    frame = frame.rename(
        columns={
            "Song": "sample_id",
            "Quadrant": "quadrant",
            "Artist": "artist",
            "Title": "title",
            "Valence": "valence_mean",
            "Arousal": "arousal_mean",
        }
    )
    frame["source"] = "merge_lyrics"
    frame["lyrics_path"] = frame.apply(
        lambda row: str(
            Path("merge")
            / "MERGE_Lyrics_Complete"
            / row["quadrant"]
            / f"{row['sample_id']}.txt"
        ),
        axis=1,
    )
    frame[["valence_class", "arousal_class"]] = frame["quadrant"].apply(
        lambda value: pd.Series(label_columns(value))
    )
    output = output_dir / "merge_lyrics.csv"
    frame.to_csv(output, index=False)
    return output


def normalize_merge_audio(raw_dir: Path, output_dir: Path) -> Path:
    root = raw_dir / "merge" / "MERGE_Audio_Complete"
    metadata = pd.read_csv(root / "merge_audio_complete_metadata.csv")
    values = pd.read_csv(root / "merge_audio_complete_av_values.csv")
    frame = metadata[["Song", "Quadrant", "Artist", "Title"]].merge(values, on="Song")
    frame = frame.rename(
        columns={
            "Song": "sample_id",
            "Quadrant": "quadrant",
            "Artist": "artist",
            "Title": "title",
            "Valence": "valence_mean",
            "Arousal": "arousal_mean",
        }
    )
    frame["source"] = "merge_audio"
    frame["audio_path"] = frame.apply(
        lambda row: str(
            Path("merge")
            / "MERGE_Audio_Complete"
            / row["quadrant"]
            / f"{row['sample_id']}.mp3"
        ),
        axis=1,
    )
    frame[["valence_class", "arousal_class"]] = frame["quadrant"].apply(
        lambda value: pd.Series(label_columns(value))
    )
    output = output_dir / "merge_audio.csv"
    frame.to_csv(output, index=False)
    return output


def normalize_merge_bimodal(raw_dir: Path, output_dir: Path) -> Path:
    root = raw_dir / "merge" / "MERGE_Bimodal_Complete"
    metadata = pd.read_csv(root / "merge_bimodal_complete_metadata.csv")
    values = pd.read_csv(root / "merge_bimodal_complete_av_values.csv")
    frame = metadata[["Audio_Song", "Lyric_Song", "Quadrant", "Artist", "Title"]]
    frame = frame.merge(values, on=["Audio_Song", "Lyric_Song"])
    frame = frame.rename(
        columns={
            "Audio_Song": "sample_id",
            "Lyric_Song": "lyric_id",
            "Quadrant": "quadrant",
            "Artist": "artist",
            "Title": "title",
            "Valence": "valence_mean",
            "Arousal": "arousal_mean",
        }
    )
    frame["source"] = "merge_bimodal"
    frame["audio_path"] = frame.apply(
        lambda row: str(
            Path("merge")
            / "MERGE_Bimodal_Complete"
            / "audio"
            / row["quadrant"]
            / f"{row['sample_id']}.mp3"
        ),
        axis=1,
    )
    frame["lyrics_path"] = frame.apply(
        lambda row: str(
            Path("merge")
            / "MERGE_Bimodal_Complete"
            / "lyrics"
            / row["quadrant"]
            / f"{row['lyric_id']}.txt"
        ),
        axis=1,
    )
    frame[["valence_class", "arousal_class"]] = frame["quadrant"].apply(
        lambda value: pd.Series(label_columns(value))
    )
    output = output_dir / "merge_bimodal.csv"
    frame.to_csv(output, index=False)
    return output


def normalize_all(raw_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        normalize_emopia(raw_dir, output_dir),
        normalize_deam(raw_dir, output_dir),
        normalize_merge_lyrics(raw_dir, output_dir),
        normalize_merge_audio(raw_dir, output_dir),
        normalize_merge_bimodal(raw_dir, output_dir),
    ]
    for output in outputs:
        print(f"normalized: {output}")
    return outputs


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    normalize_all(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
