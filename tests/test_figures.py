import unittest

from music_emotion.figures import best_per_modality, lyrics_arc, render

ROWS = [
    {"modality": "symbolic", "model": "LogReg", "macro_f1": "0.632"},
    {"modality": "symbolic", "model": "LogReg balanced", "macro_f1": "0.639"},
    {"modality": "lyrics", "model": "LinearSVC tuned (nested CV)", "macro_f1": "0.690"},
    {
        "modality": "transformer",
        "model": "LinearSVC reference (official split)",
        "macro_f1": "0.665",
    },
    {"modality": "transformer", "model": "DistilBERT fine-tuned (3 epochs)", "macro_f1": "0.716"},
    {"modality": "llm", "model": "gpt-5.5 (zero-shot)", "macro_f1": "0.764"},
]


class FigureTests(unittest.TestCase):
    def test_best_per_modality_picks_max(self) -> None:
        best = dict((mod, score) for mod, _, score in best_per_modality(ROWS))
        self.assertAlmostEqual(best["symbolic"], 0.639)
        self.assertAlmostEqual(best["llm"], 0.764)

    def test_best_per_modality_is_ordered(self) -> None:
        order = [mod for mod, _, _ in best_per_modality(ROWS)]
        self.assertEqual(order, ["symbolic", "lyrics", "transformer", "llm"])

    def test_lyrics_arc_is_monotonic(self) -> None:
        scores = [score for _, score in lyrics_arc(ROWS)]
        self.assertEqual(scores, sorted(scores))

    def test_render_produces_svg(self) -> None:
        figures = render(ROWS)
        for name, svg in figures.items():
            self.assertTrue(name.endswith(".svg"))
            self.assertTrue(svg.startswith("<svg"))
            self.assertTrue(svg.endswith("</svg>"))

    def test_render_is_deterministic(self) -> None:
        self.assertEqual(render(ROWS), render(ROWS))


if __name__ == "__main__":
    unittest.main()
