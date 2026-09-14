"""Render blog figures straight from the logged results.

Pure stdlib SVG so the charts stay reproducible in CI without a plotting stack.
"""

from __future__ import annotations

import csv
from argparse import ArgumentParser
from pathlib import Path

RESULTS = Path("results/results_summary.csv")
FIGURES = Path("results/figures")

INK = "#1c1c1c"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"
BAR = "#4a6fa5"
BAR_WEAK = "#b8c4d6"
HIGHLIGHT = "#c2643a"


def load_rows(path: Path = RESULTS) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def best_per_modality(rows: list[dict[str, str]]) -> list[tuple[str, str, float]]:
    """Highest macro-F1 per modality, in pipeline order."""
    order = ["symbolic", "audio", "lyrics", "transformer", "fusion", "llm"]
    best: dict[str, tuple[str, float]] = {}
    for row in rows:
        score = float(row["macro_f1"])
        current = best.get(row["modality"])
        if current is None or score > current[1]:
            best[row["modality"]] = (row["model"], score)
    return [(m, best[m][0], best[m][1]) for m in order if m in best]


def lyrics_arc(rows: list[dict[str, str]]) -> list[tuple[str, float]]:
    """The TF-IDF -> fine-tuned -> zero-shot progression on lyrics."""
    wanted = [
        ("transformer", "LinearSVC reference (official split)", "TF-IDF SVM"),
        ("transformer", "DistilBERT fine-tuned (3 epochs)", "DistilBERT\nfine-tuned"),
        ("llm", "gpt-5.5 (zero-shot)", "GPT-5.5\nzero-shot"),
    ]
    index = {(r["modality"], r["model"]): float(r["macro_f1"]) for r in rows}
    return [(label, index[(mod, name)]) for mod, name, label in wanted]


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _bar_chart(
    title: str,
    subtitle: str,
    bars: list[tuple[str, float]],
    highlight: int | None = None,
    width: int = 820,
) -> str:
    left, right, top, bottom = 250, 70, 64, 40
    row_height = 42
    height = top + row_height * len(bars) + bottom
    plot_width = width - left - right
    ceiling = 0.8

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui, -apple-system, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-size="16" font-weight="600" fill="{INK}">'
        f"{_escape(title)}</text>",
        f'<text x="{left}" y="45" font-size="12" fill="{MUTED}">{_escape(subtitle)}</text>',
    ]

    for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        x = left + plot_width * (tick / ceiling)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + row_height * len(bars)}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + row_height * len(bars) + 20}" font-size="11" '
            f'fill="{MUTED}" text-anchor="middle">{tick:.1f}</text>'
        )

    for i, (label, value) in enumerate(bars):
        y = top + i * row_height + 8
        bar_width = plot_width * (value / ceiling)
        color = HIGHLIGHT if highlight == i else BAR
        lines = label.split("\n")
        offset = 20 - (len(lines) - 1) * 6
        for j, line in enumerate(lines):
            parts.append(
                f'<text x="{left - 12}" y="{y + offset + j * 13}" font-size="12" fill="{INK}" '
                f'text-anchor="end">{_escape(line)}</text>'
            )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="26" rx="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + bar_width + 8:.1f}" y="{y + 18}" font-size="12" '
            f'font-weight="600" fill="{INK}">{value:.3f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render(rows: list[dict[str, str]]) -> dict[str, str]:
    modality = best_per_modality(rows)
    modality_bars = [(f"{mod}\n{model}", score) for mod, model, score in modality]
    best_index = max(range(len(modality_bars)), key=lambda i: modality_bars[i][1])

    arc = lyrics_arc(rows)

    return {
        "best_per_modality.svg": _bar_chart(
            "Best macro-F1 by modality",
            "Each bar is the strongest configuration for that input type",
            modality_bars,
            highlight=best_index,
        ),
        "lyrics_arc.svg": _bar_chart(
            "Three eras of lyrics classification",
            "Same task, same held-out split: feature engineering to fine-tuning to zero-shot",
            arc,
            highlight=len(arc) - 1,
        ),
    }


def write(output_dir: Path = FIGURES, results_path: Path = RESULTS) -> list[Path]:
    figures = render(load_rows(results_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, svg in figures.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        written.append(path)
        print(f"wrote {path}")
    return written


def check(output_dir: Path = FIGURES, results_path: Path = RESULTS) -> int:
    """Fail if the committed figures drift from the logged results."""
    figures = render(load_rows(results_path))
    stale = []
    for name, svg in figures.items():
        path = output_dir / name
        if not path.exists() or path.read_text(encoding="utf-8") != svg:
            stale.append(name)
    if stale:
        print("figures are stale, rerun `python -m music_emotion.figures`: " + ", ".join(stale))
        return 1
    print(f"figures up to date ({len(figures)})")
    return 0


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    parser.add_argument(
        "--check", action="store_true", help="verify committed figures match the results"
    )
    args = parser.parse_args()
    if args.check:
        raise SystemExit(check(args.output_dir, args.results))
    write(args.output_dir, args.results)


if __name__ == "__main__":
    main()
