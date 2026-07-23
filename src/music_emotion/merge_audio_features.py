from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE = 22050
DURATION = 30.0


def clip_features(path: Path) -> dict[str, float]:
    import librosa

    signal, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    if signal.size == 0:
        raise ValueError(f"empty audio: {path}")

    row: dict[str, float] = {}

    def summarise(name: str, values: np.ndarray) -> None:
        values = np.atleast_2d(values)
        for i, series in enumerate(values):
            suffix = f"{name}_{i}" if values.shape[0] > 1 else name
            row[f"{suffix}_mean"] = float(np.mean(series))
            row[f"{suffix}_std"] = float(np.std(series))

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    summarise("mfcc", mfcc)
    summarise("mfcc_delta", librosa.feature.delta(mfcc))
    summarise("chroma", librosa.feature.chroma_stft(y=signal, sr=sr))
    summarise("contrast", librosa.feature.spectral_contrast(y=signal, sr=sr))
    summarise("centroid", librosa.feature.spectral_centroid(y=signal, sr=sr))
    summarise("bandwidth", librosa.feature.spectral_bandwidth(y=signal, sr=sr))
    summarise("rolloff", librosa.feature.spectral_rolloff(y=signal, sr=sr))
    summarise("zcr", librosa.feature.zero_crossing_rate(signal))
    summarise("rms", librosa.feature.rms(y=signal))

    tempo = librosa.beat.tempo(y=signal, sr=sr)
    row["tempo"] = float(tempo[0]) if len(tempo) else 0.0
    return row


def aggregate_merge_bimodal(raw_dir: Path, processed_dir: Path) -> Path:
    labels = pd.read_csv(processed_dir / "merge_bimodal.csv")
    rows = []
    total = len(labels)
    for index, record in enumerate(labels.itertuples(index=False), start=1):
        path = raw_dir / record.audio_path
        try:
            features = clip_features(path)
        except Exception as error:
            print(f"skip {path.name}: {error}")
            continue
        features["sample_id"] = record.sample_id
        rows.append(features)
        if index % 100 == 0 or index == total:
            print(f"aggregate merge audio: {index}/{total}")

    features_frame = pd.DataFrame(rows)
    output = processed_dir / "merge_bimodal_audio_features.csv"
    features_frame.to_csv(output, index=False)
    print(f"saved: {output} ({len(features_frame)} clips, {features_frame.shape[1] - 1} features)")
    return output


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    aggregate_merge_bimodal(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
