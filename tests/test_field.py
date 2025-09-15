import unittest

import numpy as np

from anaspike.dataclasses.field import GridField2D, calculate_psd_2d



class CalculatePsd2dTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

class Test2dSineWave(CalculatePsd2dTestCase):
    def setUp(self):
        from anaspike.dataclasses.interval import Interval
        from anaspike.dataclasses.grid import RegularGrid1D, RegularGrid2D
        self.fx = 5.0
        self.fy = 2.0
        grid = RegularGrid2D(RegularGrid1D.given_n(Interval(-0.3, 0.7), 1000),
                             RegularGrid1D.given_n(Interval(0, 0.5), 1000))
        zz = np.sin(2 * np.pi * (self.fx * grid.xx + self.fy * grid.yy))
        self.signal = GridField2D(grid, zz)

    def test(self):
        psd = calculate_psd_2d(self.signal)
        peak_frequency_idx = np.unravel_index(np.argmax(psd.elements),
                                              psd.elements.shape)
        peak_x_frequency = psd.grid.x.points[peak_frequency_idx[0]]
        peak_y_frequency = psd.grid.y.points[peak_frequency_idx[1]]

        self.assertAlmostEqual(self.fx, abs(peak_x_frequency))
        self.assertAlmostEqual(self.fy, abs(peak_y_frequency))

