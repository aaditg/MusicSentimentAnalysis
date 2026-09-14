import unittest

from music_emotion.timeline import Window, circumplex, format_timeline, render_svg


def make_window(start: float, quadrant: str, scores: dict[str, float]) -> Window:
    valence, arousal = circumplex(scores)
    return Window(start, start + 30.0, quadrant, valence, arousal, scores)


class CircumplexTests(unittest.TestCase):
    def test_joyful_is_positive_on_both_axes(self) -> None:
        valence, arousal = circumplex({"Q1": 2.0, "Q2": -1.0, "Q3": -1.0, "Q4": 0.0})
        self.assertGreater(valence, 0)
        self.assertGreater(arousal, 0)

    def test_gloomy_is_negative_on_both_axes(self) -> None:
        valence, arousal = circumplex({"Q1": -1.0, "Q2": 0.0, "Q3": 2.0, "Q4": -1.0})
        self.assertLess(valence, 0)
        self.assertLess(arousal, 0)

    def test_calm_is_positive_valence_negative_arousal(self) -> None:
        valence, arousal = circumplex({"Q1": 0.0, "Q2": -1.0, "Q3": -1.0, "Q4": 2.0})
        self.assertGreater(valence, 0)
        self.assertLess(arousal, 0)

    def test_tense_is_negative_valence_positive_arousal(self) -> None:
        valence, arousal = circumplex({"Q1": 0.0, "Q2": 2.0, "Q3": -1.0, "Q4": -1.0})
        self.assertLess(valence, 0)
        self.assertGreater(arousal, 0)

    def test_flat_scores_sit_at_the_origin(self) -> None:
        valence, arousal = circumplex({"Q1": 1.0, "Q2": 1.0, "Q3": 1.0, "Q4": 1.0})
        self.assertAlmostEqual(valence, 0.0)
        self.assertAlmostEqual(arousal, 0.0)


class TimelineOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.windows = [
            make_window(0.0, "Q4", {"Q1": 0.0, "Q2": -1.0, "Q3": -1.0, "Q4": 2.0}),
            make_window(30.0, "Q3", {"Q1": -1.0, "Q2": 0.0, "Q3": 2.0, "Q4": -1.0}),
            make_window(60.0, "Q3", {"Q1": -1.0, "Q2": 0.0, "Q3": 2.0, "Q4": -1.0}),
        ]

    def test_format_reports_dominant_quadrant(self) -> None:
        text = format_timeline(self.windows, "Test")
        self.assertIn("Dominant: Q3", text)
        self.assertIn("2/3", text)
        self.assertIn("0:30-1:00", text)

    def test_svg_is_well_formed(self) -> None:
        svg = render_svg(self.windows, "Test", "subtitle")
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertIn("valence", svg)
        self.assertIn("arousal", svg)

    def test_svg_handles_a_single_window(self) -> None:
        svg = render_svg(self.windows[:1], "Test", "subtitle")
        self.assertTrue(svg.endswith("</svg>"))


if __name__ == "__main__":
    unittest.main()
