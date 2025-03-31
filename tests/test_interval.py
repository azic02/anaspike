import unittest

from anaspike.dataclasses.interval import Interval



class TestInterval(unittest.TestCase):

    def setUp(self):
        self.interval = Interval(0, 10)

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

    def test_bin_given_num(self):
        result = self.interval.bin(5)
        expected = [Interval(0, 2), Interval(2, 4), Interval(4, 6), Interval(6, 8), Interval(8, 10)]

        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertEqual(r.start, e.start)
            self.assertEqual(r.end, e.end)

    def test_bin_given_size(self):
        result = self.interval.bin(size=2)
        expected = [Interval(0, 2), Interval(2, 4), Interval(4, 6), Interval(6, 8)]

        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertEqual(r.start, e.start)
            self.assertEqual(r.end, e.end)

    def test_bin_given_size_float(self):
        result = Interval(100.,10000.).bin(size=500.)
        expected = [Interval(100., 600.), Interval(600., 1100.), Interval(1100., 1600.), Interval(1600., 2100.),
                    Interval(2100., 2600.), Interval(2600., 3100.), Interval(3100., 3600.), Interval(3600., 4100.),
                    Interval(4100., 4600.), Interval(4600., 5100.), Interval(5100., 5600.), Interval(5600., 6100.),
                    Interval(6100., 6600.), Interval(6600., 7100.), Interval(7100., 7600.), Interval(7600., 8100.),
                    Interval(8100., 8600.), Interval(8600., 9100.), Interval(9100., 9600.),
                    ]

        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertEqual(r.start, e.start)
            self.assertEqual(r.end, e.end)


if __name__ == '__main__':
    unittest.main(verbosity=2)

