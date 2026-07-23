QUADRANTS = {
    "Q1": ("high", "high"),
    "Q2": ("low", "high"),
    "Q3": ("low", "low"),
    "Q4": ("high", "low"),
}


def quadrant_from_values(valence: float, arousal: float, midpoint: float) -> str:
    if valence >= midpoint and arousal >= midpoint:
        return "Q1"
    if valence < midpoint and arousal >= midpoint:
        return "Q2"
    if valence < midpoint and arousal < midpoint:
        return "Q3"
    return "Q4"


def label_columns(quadrant: str) -> tuple[str, str]:
    return QUADRANTS[quadrant]
