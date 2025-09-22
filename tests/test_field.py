import unittest

import numpy as np

from anaspike.dataclasses.interval import Interval
from anaspike.dataclasses.grid import RegularGrid1D, RegularGrid2D
from anaspike.dataclasses.field import (GridField2D,
                                        calculate_fft_2d,
                                        calculate_ifft_2d,
                                        calculate_psd_2d,
                                        calculate_autocorrelation_2d_wiener_khinchin,
                                        )



class TestGridField2DConstruction(unittest.TestCase):
    def test_incompatible_number_of_values(self):
        nx = 3
        ny = 4
        x_coords = np.linspace(0., 2., nx, dtype=np.float64)
        y_coords = np.linspace(-1., 2., ny, dtype=np.float64)
        grid = RegularGrid2D(RegularGrid1D(x_coords), RegularGrid1D(y_coords))
        values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        with self.assertRaises(ValueError):
            GridField2D(grid, values)

    def test_incompatible_value_array_shape(self):
        nx = 3
        ny = 4
        x_coords = np.linspace(0., 2., nx, dtype=np.float64)
        y_coords = np.linspace(-1., 2., ny, dtype=np.float64)
        grid = RegularGrid2D(RegularGrid1D(x_coords), RegularGrid1D(y_coords))
        values = np.array([[0, 1, 2],
                           [4, 5, 6],
                           [8, 9, 10],
                           [12, 13, 14]])
        with self.assertRaises(ValueError):
            GridField2D(grid, values)

class TestGridField2DCorrectCoords(unittest.TestCase):
    def setUp(self):
        self.nx = 3
        self.ny = 4
        self.x_coords = np.linspace(0., 2., self.nx, dtype=np.float64)
        self.y_coords = np.linspace(-1., 2., self.ny, dtype=np.float64)
        grid = RegularGrid2D(RegularGrid1D(self.x_coords), RegularGrid1D(self.y_coords))
        self.values = np.array([[0, 1, 2, 3],
                                [4, 5, 6, 7],
                                [8, 9, 10, 11]])
        self.gf = GridField2D(grid, self.values)

    def test(self):
        for i in range(self.nx):
            for j in range(self.ny):
                self.assertAlmostEqual(self.gf.xx[i, j], self.x_coords[i])
                self.assertAlmostEqual(self.gf.yy[i, j], self.y_coords[j])
                self.assertAlmostEqual(self.gf.elements[i, j], self.values[i, j])

class TestFFT(unittest.TestCase):
    def test_shape(self):
        nx = 8
        ny = 7
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), nx, endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), ny, endpoint=False))
        values = np.random.rand(nx, ny)
        gf = GridField2D(grid, values)
        fft_gf = calculate_fft_2d(gf)
        self.assertEqual(fft_gf.elements.shape, (nx, ny))
        self.assertEqual(fft_gf.grid.shape, (nx, ny))

    def test_origin_at_center(self):
        nx = 8
        ny = 7
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), nx, endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), ny, endpoint=False))
        values = np.random.rand(nx, ny)
        gf = GridField2D(grid, values)
        fft_gf = calculate_fft_2d(gf)
        center_idx = (fft_gf.nx // 2, fft_gf.ny // 2)
        self.assertAlmostEqual(0.0, fft_gf.xx[center_idx])
        self.assertAlmostEqual(0.0, fft_gf.yy[center_idx])

class TestIFFT(unittest.TestCase):
    def test_shape(self):
        nx = 8
        ny = 7
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), nx, endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), ny, endpoint=False))
        values = np.random.rand(nx, ny)
        gf = GridField2D(grid, values)
        fft = calculate_fft_2d(gf)
        ifft = calculate_ifft_2d(fft)
        self.assertEqual(ifft.elements.shape, (nx, ny))
        self.assertEqual(ifft.grid.shape, (nx, ny))

    def test_origin_at_center(self):
        nx = 8
        ny = 7
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), nx, endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), ny, endpoint=False))
        values = np.random.rand(nx, ny)
        gf = GridField2D(grid, values)
        fft = calculate_fft_2d(gf)
        ifft = calculate_ifft_2d(fft)
        center_idx = (ifft.nx // 2, ifft.ny // 2)
        self.assertAlmostEqual(0.0, ifft.xx[center_idx])
        self.assertAlmostEqual(0.0, ifft.yy[center_idx])

    def test_ifft_sine_wave_reconstruction(self):
        fx = 10.0
        fy = 5.0
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(-0.5, 0.5),
                                                             1000,
                                                    endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(-0.25, 0.25),
                                                    1000,
                                                    endpoint=False))
        signal = GridField2D(grid, np.sin(2 * np.pi * (fx * grid.xx + fy * grid.yy)))
        fft = calculate_fft_2d(signal)
        ifft = calculate_ifft_2d(fft)
        np.testing.assert_array_almost_equal(ifft.xx, signal.xx)
        np.testing.assert_array_almost_equal(ifft.yy, signal.yy)
        np.testing.assert_array_almost_equal(ifft.elements, signal.elements)


class TestPSD(unittest.TestCase):
    def test_correct_shape(self):
        nx = 8
        ny = 7
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), nx, endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), ny, endpoint=False))
        values = np.random.rand(nx, ny)
        gf = GridField2D(grid, values)
        psd = calculate_psd_2d(gf)
        self.assertEqual(psd.elements.shape, (nx, ny))
        self.assertEqual(psd.grid.shape, (nx, ny))

    def test_peak_at_origin_for_constant_signal(self):
        nx = 8
        ny = 7
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), nx, endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), ny, endpoint=False))
        values = np.ones((nx, ny))
        gf = GridField2D(grid, values)
        psd = calculate_psd_2d(gf)
        peak_idx = gf.unravel_index(np.argmax(psd.elements))
        self.assertEqual((nx // 2, ny // 2), peak_idx)
        self.assertAlmostEqual(0.0, psd.xx[peak_idx])
        self.assertAlmostEqual(0.0, psd.yy[peak_idx])

    def test_correct_peak_frequency_for_2d_sine_wave(self):
        fx = 10.0
        fy = 5.0
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0, 1),
                                                             1000,
                                                    endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(-1, 2),
                                                    1000,
                                                    endpoint=False))
        gf = GridField2D(grid, np.sin(2 * np.pi * (fx * grid.xx + fy * grid.yy)))
        psd = calculate_psd_2d(gf)
        peak_indices = gf.unravel_index(np.argmax(psd.elements))
        peak_x_frequency = psd.xx[peak_indices]
        peak_y_frequency = psd.yy[peak_indices]
        self.assertAlmostEqual(fx, abs(peak_x_frequency))
        self.assertAlmostEqual(fy, abs(peak_y_frequency))


class TestAutocorrelation(unittest.TestCase):
    def test_correct_shape(self):
        nx = 8
        ny = 7
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), nx, endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), ny, endpoint=False))
        values = np.random.rand(nx, ny)
        gf = GridField2D(grid, values)
        from anaspike.dataclasses.field import calculate_autocorrelation_2d_wiener_khinchin
        ac = calculate_autocorrelation_2d_wiener_khinchin(gf)
        self.assertEqual(ac.elements.shape, (nx, ny))
        self.assertEqual(ac.grid.shape, (nx, ny))

    def test_origin_at_center(self):
        nx = 8
        ny = 7
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), nx, endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), ny, endpoint=False))
        values = np.random.rand(nx, ny)
        gf = GridField2D(grid, values)
        ac = calculate_autocorrelation_2d_wiener_khinchin(gf)
        center_idx = (ac.nx // 2, ac.ny // 2)
        self.assertAlmostEqual(0.0, ac.xx[center_idx])
        self.assertAlmostEqual(0.0, ac.yy[center_idx])

    def test_peak_at_origin(self):
        nx = 8
        ny = 7
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), nx, endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), ny, endpoint=False))
        values = np.random.rand(nx, ny)
        gf = GridField2D(grid, values)
        ac = calculate_autocorrelation_2d_wiener_khinchin(gf)
        center_idx = (ac.nx // 2, ac.ny // 2)
        peak_idx = gf.unravel_index(np.argmax(ac.elements))
        self.assertEqual(center_idx, peak_idx)

    def test_peak_at_origin_is_1(self):
        nx = 8
        ny = 7
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), nx, endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(0., 1.), ny, endpoint=False))
        values = np.random.rand(nx, ny)
        gf = GridField2D(grid, values)
        ac = calculate_autocorrelation_2d_wiener_khinchin(gf)
        center_idx = (ac.nx // 2, ac.ny // 2)
        self.assertAlmostEqual(1.0, ac.elements[center_idx])

    def test_ac_at_periodic_intervals_is_1_for_2d_sine_wave(self):
        fx = 10.0
        fy = 5.0
        x_interval = Interval(0, 1)
        y_interval = Interval(-1, 2)
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(x_interval, 1000, endpoint=False),
                RegularGrid1D.from_interval_given_n(y_interval, 1000, endpoint=False))
        gf = GridField2D(grid, np.sin(2 * np.pi * (fx * grid.xx + fy * grid.yy)))
        ac = calculate_autocorrelation_2d_wiener_khinchin(gf)

        expected_x_period = 1.0 / fx
        expected_y_period = 1.0 / fy
        expected_peaks_xs = np.concatenate((-np.arange(expected_x_period, x_interval.width/2, expected_x_period)[::-1].T,
                                            np.arange(0, x_interval.width/2., expected_x_period)))
        expected_peaks_ys = np.concatenate((-np.arange(expected_y_period, y_interval.width/2, expected_y_period)[::-1],
                                            np.arange(0, y_interval.width/2., expected_y_period)))
        origin_idx = (ac.nx // 2, ac.ny // 2)
        expected_peak_x_idxs = np.round(expected_peaks_xs / ac.grid.delta_x).astype(int) + origin_idx[0]
        expected_peak_y_idxs = np.round(expected_peaks_ys / ac.grid.delta_y).astype(int) + origin_idx[1]
        from itertools import product
        expected_peak_idxs = np.array(list(product(expected_peak_x_idxs, expected_peak_y_idxs)))
        values_at_expected_peaks = ac.elements[expected_peak_idxs[:,0], expected_peak_idxs[:,1]]
        np.testing.assert_allclose(values_at_expected_peaks, 1.0, atol=5e-4, rtol=5e-4)

    def test_autocorrelation_of_autocorrelation_is_autocorrelation(self):
        fx = 10.0
        fy = 5.0
        grid = RegularGrid2D(
                RegularGrid1D.from_interval_given_n(Interval(0, 1),
                                                             1000,
                                                    endpoint=False),
                RegularGrid1D.from_interval_given_n(Interval(-1, 2),
                                                    1000,
                                                    endpoint=False))
        gf = GridField2D(grid, np.sin(2 * np.pi * (fx * grid.xx + fy * grid.yy)))
        ac = calculate_autocorrelation_2d_wiener_khinchin(gf)
        ac_of_ac = calculate_autocorrelation_2d_wiener_khinchin(ac)
        np.testing.assert_array_almost_equal(ac.elements, ac_of_ac.elements)

