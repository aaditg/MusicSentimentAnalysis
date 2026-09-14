import unittest

from music_emotion.timeline import (
    Window,
    _axis_limit,
    circumplex,
    format_timeline,
    render_svg,
)


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


class CalibratedTests(unittest.TestCase):
    def probs(self, q1, q2, q3, q4):
        return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4}

    def test_probabilities_stay_within_unit_range(self) -> None:
        for scores in (
            self.probs(1.0, 0.0, 0.0, 0.0),
            self.probs(0.0, 0.0, 1.0, 0.0),
            self.probs(0.25, 0.25, 0.25, 0.25),
            self.probs(0.4, 0.1, 0.2, 0.3),
        ):
            valence, arousal = circumplex(scores, probabilities=True)
            self.assertGreaterEqual(valence, -1.0)
            self.assertLessEqual(valence, 1.0)
            self.assertGreaterEqual(arousal, -1.0)
            self.assertLessEqual(arousal, 1.0)

    def test_certain_prediction_reaches_the_corner(self) -> None:
        valence, arousal = circumplex(self.probs(1.0, 0.0, 0.0, 0.0), probabilities=True)
        self.assertAlmostEqual(valence, 1.0)
        self.assertAlmostEqual(arousal, 1.0)

    def test_uniform_probabilities_sit_at_the_origin(self) -> None:
        valence, arousal = circumplex(self.probs(0.25, 0.25, 0.25, 0.25), probabilities=True)
        self.assertAlmostEqual(valence, 0.0)
        self.assertAlmostEqual(arousal, 0.0)

    def test_confidence_is_the_winning_probability(self) -> None:
        scores = self.probs(0.55, 0.15, 0.2, 0.1)
        window = Window(0.0, 30.0, "Q1", 0.0, 0.0, scores, calibrated=True)
        self.assertAlmostEqual(window.confidence, 0.55)

    def test_confidence_is_none_when_uncalibrated(self) -> None:
        window = Window(0.0, 30.0, "Q1", 0.0, 0.0, self.probs(2.0, -1.0, 0.0, 1.0))
        self.assertIsNone(window.confidence)

    def test_calibrated_axis_is_fixed(self) -> None:
        small = [Window(0.0, 30.0, "Q1", 0.1, 0.05, self.probs(0.4, 0.2, 0.2, 0.2), True)]
        self.assertEqual(_axis_limit(small), 1.0)

    def test_uncalibrated_axis_scales_to_data(self) -> None:
        windows = [Window(0.0, 30.0, "Q1", 2.0, 1.0, self.probs(2.0, 0.0, 0.0, 1.0))]
        self.assertGreater(_axis_limit(windows), 2.0)

    def test_format_shows_confidence_only_when_calibrated(self) -> None:
        scores = self.probs(0.6, 0.1, 0.2, 0.1)
        calibrated = [Window(0.0, 30.0, "Q1", 0.4, 0.4, scores, calibrated=True)]
        self.assertIn("conf", format_timeline(calibrated, "T"))
        self.assertIn("60%", format_timeline(calibrated, "T"))
        raw = [Window(0.0, 30.0, "Q1", 0.4, 0.4, scores)]
        self.assertNotIn("conf", format_timeline(raw, "T"))

    def test_svg_renders_calibrated_windows(self) -> None:
        scores = self.probs(0.6, 0.1, 0.2, 0.1)
        windows = [
            Window(0.0, 30.0, "Q1", 0.4, 0.4, scores, calibrated=True),
            Window(30.0, 60.0, "Q3", -0.4, -0.4, scores, calibrated=True),
        ]
        svg = render_svg(windows, "T", "s")
        self.assertTrue(svg.endswith("</svg>"))


if __name__ == "__main__":
    unittest.main()
