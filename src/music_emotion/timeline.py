"""Score a song window by window and chart how its emotion moves.

The training annotations (DEAM, MERGE) describe 30-second excerpts, so a
30-second window is the granularity the model was actually fitted on. Sliding
that window across a whole track keeps every prediction in-distribution and
turns a single label into a trajectory.

Valence and arousal are recovered by projecting the four quadrant scores back
onto the circumplex axes they came from:

    valence = (Q1 + Q4) - (Q2 + Q3)
    arousal = (Q1 + Q2) - (Q3 + Q4)
"""

from __future__ import annotations

import csv
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from .figures import INK, MUTED, _escape
from .merge_audio_features import SAMPLE_RATE, summarise_signal
from .predict import MODEL_PATH

WINDOW = 30.0
HOP = 30.0

QUADRANT_COLORS = {
    "Q1": "#d9a441",  # joyful
    "Q2": "#c2643a",  # tense
    "Q3": "#4a6fa5",  # gloomy
    "Q4": "#5c9e78",  # calm
}
QUADRANT_NAMES = {
    "Q1": "joyful",
    "Q2": "tense",
    "Q3": "gloomy",
    "Q4": "calm",
}


@dataclass
class Window:
    start: float
    end: float
    quadrant: str
    valence: float
    arousal: float
    scores: dict[str, float]

    @property
    def label(self) -> str:
        return f"{self.quadrant} ({QUADRANT_NAMES[self.quadrant]})"


def _timestamp(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def window_features(
    audio_path: Path, window: float = WINDOW, hop: float = HOP
) -> list[tuple[float, float, dict[str, float]]]:
    """Feature rows for each window, using the same extractor as training."""
    import librosa

    signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if signal.size == 0:
        raise ValueError(f"empty audio: {audio_path}")

    span = int(window * sr)
    step = int(hop * sr)
    rows = []
    for start in range(0, max(signal.size - span // 2, 1), step):
        chunk = signal[start : start + span]
        if chunk.size < span // 2:
            break
        rows.append((start / sr, (start + chunk.size) / sr, summarise_signal(chunk, sr)))
    return rows


def circumplex(scores: dict[str, float]) -> tuple[float, float]:
    valence = (scores["Q1"] + scores["Q4"]) - (scores["Q2"] + scores["Q3"])
    arousal = (scores["Q1"] + scores["Q2"]) - (scores["Q3"] + scores["Q4"])
    return valence / 2.0, arousal / 2.0


def analyse(
    audio_path: Path,
    lyrics_path: Path | None = None,
    model_path: Path = MODEL_PATH,
    window: float = WINDOW,
    hop: float = HOP,
) -> list[Window]:
    import joblib
    import numpy as np
    import pandas as pd

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} is missing - run `python -m music_emotion.predict train` first"
        )
    bundle = joblib.load(model_path)
    model, audio_columns = bundle["model"], bundle["audio_columns"]

    windows = window_features(audio_path, window, hop)
    if not windows:
        raise ValueError(f"audio is shorter than one {window:g}s window: {audio_path}")

    frame = pd.DataFrame([features for _, _, features in windows]).reindex(columns=audio_columns)

    if lyrics_path is None:
        # Audio branch only: no lyrics means no text features to stand on.
        scorer = model.audio_
        frame_for_model = frame
    else:
        scorer = model
        frame_for_model = frame.copy()
        frame_for_model["text"] = lyrics_path.read_text(encoding="utf-8", errors="replace")

    matrix = scorer.decision_function(frame_for_model)
    classes = [str(c) for c in model.classes_]

    results = []
    for (start, end, _), row in zip(windows, matrix, strict=True):
        scores = {c: float(s) for c, s in zip(classes, row, strict=True)}
        valence, arousal = circumplex(scores)
        results.append(
            Window(
                start=start,
                end=end,
                quadrant=classes[int(np.argmax(row))],
                valence=valence,
                arousal=arousal,
                scores=scores,
            )
        )
    return results


def format_timeline(windows: list[Window], title: str) -> str:
    lines = [title, "=" * len(title), ""]
    lines.append(f"{'time':<14}{'quadrant':<16}{'valence':>9}{'arousal':>9}")
    for w in windows:
        span = f"{_timestamp(w.start)}-{_timestamp(w.end)}"
        lines.append(f"{span:<14}{w.label:<16}{w.valence:>+9.2f}{w.arousal:>+9.2f}")

    counts: dict[str, int] = {}
    for w in windows:
        counts[w.quadrant] = counts.get(w.quadrant, 0) + 1
    dominant = max(counts, key=lambda q: counts[q])
    lines += [
        "",
        f"Windows: {len(windows)}   Dominant: {dominant} ({QUADRANT_NAMES[dominant]}), "
        f"{counts[dominant]}/{len(windows)}",
    ]
    return "\n".join(lines)


def write_csv(windows: list[Window], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_s", "end_s", "quadrant", "valence", "arousal", *QUADRANT_COLORS])
        for w in windows:
            writer.writerow(
                [
                    f"{w.start:.1f}",
                    f"{w.end:.1f}",
                    w.quadrant,
                    f"{w.valence:.4f}",
                    f"{w.arousal:.4f}",
                    *[f"{w.scores[q]:.4f}" for q in QUADRANT_COLORS],
                ]
            )
    return path


def render_svg(windows: list[Window], title: str, subtitle: str, width: int = 900) -> str:
    left, right, top, bottom = 60, 110, 74, 92
    plot_height = 220
    strip_height = 26
    height = top + plot_height + strip_height + bottom
    plot_width = width - left - right

    span = windows[-1].end - windows[0].start
    limit = max(1e-6, max(max(abs(w.valence), abs(w.arousal)) for w in windows) * 1.15)

    def x_of(t: float) -> float:
        return left + plot_width * ((t - windows[0].start) / span)

    def y_of(value: float) -> float:
        return top + plot_height / 2 - (value / limit) * (plot_height / 2)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui, -apple-system, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-size="16" font-weight="600" fill="{INK}">'
        f"{_escape(title)}</text>",
        f'<text x="{left}" y="48" font-size="12" fill="{MUTED}">{_escape(subtitle)}</text>',
    ]

    zero = y_of(0.0)
    parts.append(
        f'<line x1="{left}" y1="{zero:.1f}" x2="{left + plot_width}" y2="{zero:.1f}" '
        f'stroke="{MUTED}" stroke-width="1"/>'
    )
    for label, y in (("+", top + 6), ("-", top + plot_height - 4)):
        parts.append(
            f'<text x="{left - 10}" y="{y}" font-size="11" fill="{MUTED}" '
            f'text-anchor="end">{label}</text>'
        )

    for series, color, name in (
        ([w.valence for w in windows], "#c2643a", "valence"),
        ([w.arousal for w in windows], "#4a6fa5", "arousal"),
    ):
        points = " ".join(
            f"{x_of((w.start + w.end) / 2):.1f},{y_of(v):.1f}"
            for w, v in zip(windows, series, strict=True)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        last_x = x_of((windows[-1].start + windows[-1].end) / 2)
        parts.append(
            f'<text x="{last_x + 10:.1f}" y="{y_of(series[-1]) + 4:.1f}" font-size="12" '
            f'font-weight="600" fill="{color}">{name}</text>'
        )

    strip_y = top + plot_height + 18
    for w in windows:
        x0, x1 = x_of(w.start), x_of(w.end)
        parts.append(
            f'<rect x="{x0:.1f}" y="{strip_y}" width="{max(x1 - x0 - 1, 1):.1f}" '
            f'height="{strip_height}" fill="{QUADRANT_COLORS[w.quadrant]}" rx="2"/>'
        )

    axis_y = strip_y + strip_height + 18
    step = max(1, len(windows) // 8)
    for w in windows[::step]:
        parts.append(
            f'<text x="{x_of(w.start):.1f}" y="{axis_y}" font-size="11" fill="{MUTED}" '
            f'text-anchor="middle">{_timestamp(w.start)}</text>'
        )

    legend_y = axis_y + 26
    x = left
    for quadrant, color in QUADRANT_COLORS.items():
        parts.append(
            f'<rect x="{x}" y="{legend_y - 10}" width="11" height="11" rx="2" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x + 16}" y="{legend_y}" font-size="11" fill="{INK}">'
            f"{quadrant} {QUADRANT_NAMES[quadrant]}</text>"
        )
        x += 110

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = ArgumentParser(description="Chart how a song's emotion develops over time")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--lyrics",
        type=Path,
        help="optional; lyrics are not time-aligned, so they shift every window equally",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--window", type=float, default=WINDOW)
    parser.add_argument("--hop", type=float, default=HOP, help="seconds between window starts")
    parser.add_argument("--out-dir", type=Path, default=Path("results/timelines"))
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    windows = analyse(args.audio, args.lyrics, args.model, args.window, args.hop)
    title = args.title or args.audio.stem
    print(format_timeline(windows, f"Emotion timeline: {title}"))

    stem = args.audio.stem.replace(" ", "_")
    csv_path = write_csv(windows, args.out_dir / f"{stem}.csv")
    source = "audio + lyrics" if args.lyrics else "audio only"
    svg = render_svg(
        windows,
        f"Emotion timeline: {title}",
        f"{args.window:g}s windows, {args.hop:g}s hop, {source}",
    )
    svg_path = args.out_dir / f"{stem}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"\nwrote {csv_path}\nwrote {svg_path}")


if __name__ == "__main__":
    main()
