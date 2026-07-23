from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")
LOG_PATH = RESULTS_DIR / "experiments.jsonl"
JOURNAL_PATH = RESULTS_DIR / "EXPERIMENTS.md"

MODALITY_ORDER = ["symbolic", "lyrics", "audio", "fusion", "transformer", "llm"]


@dataclass
class Experiment:
    modality: str
    name: str
    config: str
    metrics: dict
    notes: str = ""
    primary_metric: str = "macro_f1_mean"


def log_experiment(experiment: Experiment, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modality": experiment.modality,
        "name": experiment.name,
        "config": experiment.config,
        "primary_metric": experiment.primary_metric,
        "metrics": experiment.metrics,
        "notes": experiment.notes,
    }
    log_path = results_dir / "experiments.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    rebuild_journal(results_dir)


def _load_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _format_metric(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def rebuild_journal(results_dir: Path = RESULTS_DIR) -> None:
    log_path = results_dir / "experiments.jsonl"
    records = _load_records(log_path)
    lines = [
        "# Experiments",
        "",
        "Cross-validated results per modality. Numbers are mean +/- std across folds.",
        "",
    ]

    modalities = MODALITY_ORDER + [
        m for m in {r["modality"] for r in records} if m not in MODALITY_ORDER
    ]

    for modality in modalities:
        subset = [r for r in records if r["modality"] == modality]
        if not subset:
            continue
        lines.append(f"## {modality.capitalize()}")
        lines.append("")
        lines.append(
            "| # | Experiment | Accuracy | Macro-F1 | Balanced acc | Config |"
        )
        lines.append("|---|---|---|---|---|---|")

        best_index = None
        best_value = -1.0
        for index, record in enumerate(subset, start=1):
            metrics = record["metrics"]
            primary = record.get("primary_metric", "macro_f1_mean")
            value = metrics.get(primary, -1)
            if isinstance(value, (int, float)) and value > best_value:
                best_value = value
                best_index = index

        for index, record in enumerate(subset, start=1):
            metrics = record["metrics"]
            acc = _pair(metrics, "accuracy_mean", "accuracy_std")
            f1 = _pair(metrics, "macro_f1_mean", "macro_f1_std")
            bal = _pair(metrics, "balanced_acc_mean", "balanced_acc_std")
            star = " *(best)*" if index == best_index else ""
            lines.append(
                f"| {index} | {record['name']}{star} | {acc} | {f1} | {bal} | {record['config']} |"
            )
        lines.append("")

        for index, record in enumerate(subset, start=1):
            metrics = record["metrics"]
            per_class = metrics.get("per_class_f1")
            note = record.get("notes", "")
            detail_bits = []
            if per_class:
                detail_bits.append(
                    "per-class F1: "
                    + ", ".join(f"{k}={_format_metric(v)}" for k, v in per_class.items())
                )
            if note:
                detail_bits.append(note)
            if detail_bits:
                lines.append(f"- **{index}. {record['name']}** — " + "; ".join(detail_bits))
        lines.append("")

    journal_path = results_dir / "EXPERIMENTS.md"
    journal_path.write_text("\n".join(lines), encoding="utf-8")


def _pair(metrics: dict, mean_key: str, std_key: str) -> str:
    mean = metrics.get(mean_key)
    std = metrics.get(std_key)
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"
