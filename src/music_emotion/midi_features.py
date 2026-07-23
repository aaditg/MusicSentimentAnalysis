from argparse import ArgumentParser
from pathlib import Path

import mido
import numpy as np
import pandas as pd

DEFAULT_TEMPO = 500000


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def note_features(path: Path) -> dict[str, float]:
    midi = mido.MidiFile(path)
    ticks_per_beat = midi.ticks_per_beat or 480
    tempo = DEFAULT_TEMPO
    tempo_changes = 0

    notes: list[int] = []
    velocities: list[int] = []
    onsets: list[float] = []
    durations: list[float] = []
    active: dict[int, list[float]] = {}
    current_time = 0.0

    for message in mido.merge_tracks(midi.tracks):
        current_time += mido.tick2second(message.time, ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
            tempo_changes += 1
        if message.type == "note_on" and message.velocity > 0:
            notes.append(message.note)
            velocities.append(message.velocity)
            onsets.append(current_time)
            active.setdefault(message.note, []).append(current_time)
        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            starts = active.get(message.note, [])
            if starts:
                durations.append(current_time - starts.pop(0))

    duration = max(current_time, 0.001)
    notes_array = np.array(notes) if notes else np.array([60])
    velocities_array = np.array(velocities) if velocities else np.array([0])
    durations_array = np.array(durations) if durations else np.array([0.0])
    intervals = np.diff(notes_array)
    abs_intervals = np.abs(intervals)

    onset_sorted = np.sort(np.array(onsets)) if onsets else np.array([0.0])
    iois = np.diff(onset_sorted)
    iois_nz = iois[iois > 1e-4]

    pitch_classes = notes_array % 12
    pc_hist = np.bincount(pitch_classes, minlength=12).astype(float)
    pc_hist_norm = pc_hist / max(pc_hist.sum(), 1.0)

    polyphony = durations_array.sum() / duration
    ioi_mean = float(iois_nz.mean()) if iois_nz.size else 0.0
    note_duration_mean = float(durations_array.mean())

    features = {
        "sample_id": path.stem,
        "note_count": len(notes),
        "note_density": len(notes) / duration,
        "tempo_changes": tempo_changes,
        "pitch_mean": float(notes_array.mean()),
        "pitch_std": float(notes_array.std()),
        "pitch_range": float(notes_array.max() - notes_array.min()),
        "distinct_pitches": int(np.unique(notes_array).size),
        "distinct_pitch_classes": int(np.unique(pitch_classes).size),
        "pitch_entropy": _entropy(np.bincount(notes_array - notes_array.min())),
        "pitch_class_entropy": _entropy(pc_hist),
        "interval_abs_mean": float(abs_intervals.mean()) if abs_intervals.size else 0.0,
        "interval_signed_mean": float(intervals.mean()) if intervals.size else 0.0,
        "interval_std": float(intervals.std()) if intervals.size else 0.0,
        "max_leap": float(abs_intervals.max()) if abs_intervals.size else 0.0,
        "pct_ascending": float((intervals > 0).mean()) if intervals.size else 0.0,
        "pct_descending": float((intervals < 0).mean()) if intervals.size else 0.0,
        "pct_repeat": float((intervals == 0).mean()) if intervals.size else 0.0,
        "leap_ratio": float((abs_intervals > 2).mean()) if abs_intervals.size else 0.0,
        "pct_low": float((notes_array < 48).mean()),
        "pct_mid": float(((notes_array >= 48) & (notes_array <= 72)).mean()),
        "pct_high": float((notes_array > 72).mean()),
        "velocity_mean": float(velocities_array.mean()),
        "velocity_std": float(velocities_array.std()),
        "velocity_min": float(velocities_array.min()),
        "velocity_max": float(velocities_array.max()),
        "velocity_range": float(velocities_array.max() - velocities_array.min()),
        "note_duration_mean": note_duration_mean,
        "note_duration_std": float(durations_array.std()),
        "ioi_mean": ioi_mean,
        "ioi_std": float(iois_nz.std()) if iois_nz.size else 0.0,
        "ioi_cv": float(iois_nz.std() / ioi_mean) if ioi_mean > 0 else 0.0,
        "polyphony_mean": float(polyphony),
        "articulation": float(note_duration_mean / ioi_mean) if ioi_mean > 0 else 0.0,
    }
    for index, value in enumerate(pc_hist_norm):
        features[f"pc_{index}"] = float(value)
    return features


def aggregate_emopia(raw_dir: Path, processed_dir: Path) -> Path:
    root = raw_dir / "emopia" / "EMOPIA_2.2" / "midis"
    paths = sorted(root.glob("*.mid"))
    if not paths:
        raise FileNotFoundError("EMOPIA MIDI files are missing")

    rows = []
    for index, path in enumerate(paths, start=1):
        rows.append(note_features(path))
        if index % 100 == 0 or index == len(paths):
            print(f"aggregate midi: {index}/{len(paths)}")

    labels = pd.read_csv(processed_dir / "emopia.csv")
    frame = labels.merge(pd.DataFrame(rows), on="sample_id")
    output = processed_dir / "emopia_midi_features.csv"
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    return output


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    aggregate_emopia(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
