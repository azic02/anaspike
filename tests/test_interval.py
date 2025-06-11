import unittest

from anaspike.dataclasses.interval import Interval, Bin



class TestInterval(unittest.TestCase):

    def setUp(self):
        self.interval = Interval(0, 10)

    def test_start(self):
        self.assertEqual(self.interval.start, 0)

    def test_end(self):
        self.assertEqual(self.interval.end, 10)

    def test_width(self):
        self.assertEqual(self.interval.width, 10)

    def test_initialization_exception(self):
        with self.assertRaises(ValueError):
            Interval(10, 0)

    def test_interval_getters(self):
        self.assertEqual(self.interval.start, 0)
        self.assertEqual(self.interval.end, 10)
        self.assertEqual(self.interval.width, 10)

    def test_interval_contains(self):
        self.assertTrue(self.interval.contains(5))
        self.assertTrue(self.interval.contains(0))
        self.assertFalse(self.interval.contains(-1))
        self.assertFalse(self.interval.contains(10))

    def test_discretize_given_n(self):
        result = self.interval.discretize(6)
        expected = [0, 2, 4, 6, 8, 10]

        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertEqual(r, e)

    def test_discretize_given_size(self):
        result = self.interval.discretize(size=2)
        expected = [0, 2, 4, 6, 8]

        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertEqual(r, e)


class TestBin(unittest.TestCase):
    def setUp(self):
        self.bin = Bin(0, 10, 5)

    def test_value(self):
        self.assertEqual(self.bin.value, 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

