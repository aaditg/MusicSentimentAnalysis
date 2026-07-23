ROADMAP = [
    "1. Harmonize labels into valence, arousal, and 4-quadrant classes.",
    "2. Build symbolic and lyrics baselines first.",
    "3. Add audio baselines for full emotional signal.",
    "4. Compare audio only, lyrics only, symbolic only, and fused models.",
    "5. Add transformer models after baseline metrics are stable.",
    "6. Add LLM API benchmarks last.",
    "7. Use feature importance and ablations to find emotion pointers.",
]


def format_roadmap() -> str:
    title = "Project Roadmap"
    rule = "-" * len(title)
    body = "\n".join(ROADMAP)
    return f"{title}\n{rule}\n{body}"
