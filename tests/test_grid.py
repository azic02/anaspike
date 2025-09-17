import unittest

import numpy as np

from anaspike.dataclasses.grid import RegularGrid1D



class TestRegularGrid1dFromStrSpec(unittest.TestCase):
    def test_four_commas(self):
        with self.assertRaises(ValueError):
            RegularGrid1D.from_str("0,1,n10,extra")

    def test_two_commas(self):
        with self.assertRaises(ValueError):
            RegularGrid1D.from_str("0,1")

    def test_start_non_numeric(self):
        with self.assertRaises(ValueError):
            RegularGrid1D.from_str("a,1,n10")

    def test_end_non_numeric(self):
        with self.assertRaises(ValueError):
            RegularGrid1D.from_str("0,b,n10")

    def test_last_part_invalid_char(self):
        with self.assertRaises(ValueError):
            RegularGrid1D.from_str("0,1,x10")

    def test_last_part_missing_char(self):
        with self.assertRaises(ValueError):
            RegularGrid1D.from_str("0,1,10")

    def test_n_non_integer(self):
        with self.assertRaises(ValueError):
            RegularGrid1D.from_str("0,1,n10.5")

    def test_d_non_numeric(self):
        with self.assertRaises(ValueError):
            RegularGrid1D.from_str("0,1,dabc")

    def test_valid_n(self):
        grid = RegularGrid1D.from_str("0,1,n5")
        expected_points = np.linspace(0, 1, 5, endpoint=True, dtype=np.float64)
        np.testing.assert_array_almost_equal(grid.points, expected_points)

    def test_valid_d(self):
        grid = RegularGrid1D.from_str("0,1,d0.2")
        expected_points = np.arange(0, 1, 0.2, dtype=np.float64)
        np.testing.assert_array_almost_equal(grid.points, expected_points)

