import unittest

import numpy as np
from numpy.testing import assert_array_equal

from anaspike.dataclasses.scalar_temporal_map import (ScalarTemporalMap,
                                                      correlation,
                                                      )



class TestScalarTemporalMapProperties(unittest.TestCase):
    def test_ndim(self):
        tm = ScalarTemporalMap(
                times=np.array([0.0, 1.0, 2.0]),
                values=np.array([0.0, 1.0, 2.0]))
        self.assertEqual(tm.ndim, 0)

    def test_shape(self):
        tm = ScalarTemporalMap(
                times=np.array([0.0, 1.0, 2.0]),
                values=np.array([0.0, 1.0, 2.0]))
        assert_array_equal(tm.shape, ())


class TestTemporalCorrelation(unittest.TestCase):
    def test_mismatched_time_lengths(self):
        tm1 = ScalarTemporalMap(
                times=np.array([0.0, 1.0, 2.0]),
                values=np.array([0.0, 1.0, 2.0]))
        tm2 = ScalarTemporalMap(
                times=np.array([0.0, 1.0]),
                values=np.array([5.0, 4.0]))
        with self.assertRaises(ValueError):
            correlation(tm1, tm2)

    def test_perfect_positive_correlation(self):
        tm1 = ScalarTemporalMap(
                times=np.array([0.0, 1.0, 2.0]),
                values=np.array([0.0, 1.0, 2.0]))
        tm2 = ScalarTemporalMap(
                times=np.array([0.0, 1.0, 2.0]),
                values=np.array([5.0, 6.0, 7.0]))
        corr = correlation(tm1, tm2)
        self.assertAlmostEqual(corr, 1.0)

    def test_perfect_negative_correlation(self):
        tm1 = ScalarTemporalMap(
                times=np.array([0.0, 1.0, 2.0]),
                values=np.array([0.0, 1.0, 2.0]))
        tm2 = ScalarTemporalMap(
                times=np.array([0.0, 1.0, 2.0]),
                values=np.array([7.0, 6.0, 5.0]))
        corr = correlation(tm1, tm2)
        self.assertAlmostEqual(corr, -1.0)

    def test_symmetry_in_arguments(self):
        tm1 = ScalarTemporalMap(
                times=np.array([0.0, 1.0, 2.0]),
                values=np.array([0.0, 1.0, 2.0]))
        tm2 = ScalarTemporalMap(
                times=np.array([0.0, 1.0, 2.0]),
                values=np.array([5.0, 6.0, 7.0]))
        corr1 = correlation(tm1, tm2)
        corr2 = correlation(tm2, tm1)
        self.assertAlmostEqual(corr1, corr2)

