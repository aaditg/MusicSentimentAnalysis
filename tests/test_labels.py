import unittest

from music_emotion.labels import label_columns, quadrant_from_values


class LabelTests(unittest.TestCase):
    def test_quadrant_values(self) -> None:
        self.assertEqual(quadrant_from_values(6.0, 6.0, 5.0), "Q1")
        self.assertEqual(quadrant_from_values(4.0, 6.0, 5.0), "Q2")
        self.assertEqual(quadrant_from_values(4.0, 4.0, 5.0), "Q3")
        self.assertEqual(quadrant_from_values(6.0, 4.0, 5.0), "Q4")

    def test_quadrant_labels(self) -> None:
        self.assertEqual(label_columns("Q1"), ("high", "high"))
        self.assertEqual(label_columns("Q4"), ("high", "low"))


if __name__ == "__main__":
    unittest.main()
