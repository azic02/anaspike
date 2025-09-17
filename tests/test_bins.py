import unittest

import numpy as np



class TestContigBins2dCreation(unittest.TestCase):
    def test_with_median_labels_rectilinear_grid(self):
        from anaspike.dataclasses.bins import ContigBins2D
        from anaspike.dataclasses.grid import RectilinearGrid2D, Grid1D

        grid = RectilinearGrid2D(x=Grid1D(np.array([0., 0.1, 0.4])),
                                 y=Grid1D(np.array([0.2, 0.3, 0.5, 0.9])))

        expected_labels = [[(0.05, 0.25), (0.05, 0.4), (0.05, 0.7)],
                           [(0.25, 0.25), (0.25, 0.4), (0.25, 0.7)]]

        bins = ContigBins2D[RectilinearGrid2D].with_median_labels(grid)

        np.testing.assert_array_almost_equal(bins.labels, expected_labels)

    def test_with_median_labels_regular_grid(self):
        from anaspike.dataclasses.bins import ContigBins2D
        from anaspike.dataclasses.grid import RegularGrid2D, RegularGrid1D

        x = RegularGrid1D(np.array([0., 0.1, 0.2]))
        y = RegularGrid1D(np.array([1., 1.3, 1.6, 1.9]))

        expected_labels = [[(0.05, 1.15), (0.05, 1.45), (0.05, 1.75)],
                           [(0.15, 1.15), (0.15, 1.45), (0.15, 1.75)]]

        bins = ContigBins2D[RegularGrid2D].with_median_labels(RegularGrid2D(x,y))

        np.testing.assert_array_almost_equal(bins.labels, expected_labels)


class TestContigBins2dProperties(unittest.TestCase):
    def setUp(self):
        from anaspike.dataclasses.bins import ContigBins2D
        from anaspike.dataclasses.grid import RectilinearGrid2D, Grid1D

        self.grid = RectilinearGrid2D(x=Grid1D(np.array([0., 0.1, 0.4])),
                                 y=Grid1D(np.array([0.2, 0.3, 0.5, 0.9])))

        self.bins = ContigBins2D[RectilinearGrid2D].with_median_labels(self.grid)

    def test_x(self):
        expected_edges = [0., 0.1, 0.4]
        expected_labels = [0.05, 0.25]

        bins_x = self.bins.x

        np.testing.assert_array_almost_equal(bins_x.edges, expected_edges)
        np.testing.assert_array_almost_equal(bins_x.labels, expected_labels)

    def test_y(self):
        eypected_edges = [0.2, 0.3, 0.5, 0.9]
        eypected_labels = [0.25, 0.4, 0.7]

        bins_y = self.bins.y

        np.testing.assert_array_almost_equal(bins_y.edges, eypected_edges)
        np.testing.assert_array_almost_equal(bins_y.labels, eypected_labels)




class TestClassSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from anaspike.dataclasses.bins import ContigBins2D
        from anaspike.dataclasses.grid import RegularGrid1D, RegularGrid2D
        from anaspike.dataclasses.coords2d import Coords2D
        bin_grid = RegularGrid2D(RegularGrid1D(np.array([0, 1, 2, 3])),
                                 RegularGrid1D(np.array([0, 1, 2])))
        cls.bins = ContigBins2D[RegularGrid2D].with_median_labels(bin_grid)
        cls.coords = Coords2D(x=[0.5, 0.5, 2.5, 0.5],
                              y=[0.5, 1.5, 1.5, 1.5])
        cls.coords_outside_bins = Coords2D(x=[0.5, 0.5, 2.5, 3.5],
                                           y=[0.5, 1.5, 1.5, 1.5])

class TestAssignBins(TestClassSetup):
    def test_coords_outside_bins(self):
        from anaspike.dataclasses.bins import assign_bins_2d
        with self.assertRaises(ValueError):
            assign_bins_2d(self.bins, self.coords_outside_bins)

    def test_assign_bins_2d(self):
        from anaspike.dataclasses.bins import assign_bins_2d
        expected_x_idxs = np.array([0, 0, 2, 0])
        expected_y_idxs = np.array([0, 1, 1, 1])
        x_idxs, y_idxs = assign_bins_2d(self.bins, self.coords)
        np.testing.assert_array_equal(x_idxs, expected_x_idxs)
        np.testing.assert_array_equal(y_idxs, expected_y_idxs)

class TestCalculateBinCounts(TestClassSetup):
    def test_coords_outside_bins(self):
        from anaspike.dataclasses.bins import calculate_bin_counts_2d
        with self.assertRaises(ValueError):
            calculate_bin_counts_2d(self.bins, self.coords_outside_bins)

    def test_calculate_bin_counts_2d(self):
        from anaspike.dataclasses.bins import calculate_bin_counts_2d
        expected_counts = np.array([[1, 2],
                                    [0, 0],
                                    [0, 1]])
        bin_counts = calculate_bin_counts_2d(self.bins, self.coords)
        np.testing.assert_array_equal(bin_counts.elements, expected_counts)

class TestCalculateBinSums(TestClassSetup):
    def test_coords_outside_bins(self):
        from anaspike.dataclasses.bins import calculate_bin_sums_2d
        from anaspike.dataclasses.field import Field2D
        field = Field2D(self.coords_outside_bins,
                        np.array(np.arange(4) + 1, dtype=np.float64))
        with self.assertRaises(ValueError):
            calculate_bin_sums_2d(self.bins, field)

    def test_calculate_bin_sums_2d_scalar_field(self):
        from anaspike.dataclasses.bins import calculate_bin_sums_2d
        from anaspike.dataclasses.field import Field2D
        field = Field2D(self.coords,
                      np.array(np.arange(4) + 1, dtype=np.float64))
        expected_sums = np.array([[1, 6],
                                  [0, 0],
                                  [0, 3]])
        bin_sums = calculate_bin_sums_2d(self.bins, field)
        np.testing.assert_array_almost_equal(bin_sums.elements, expected_sums)

    def test_calculate_bin_sums_2d_vector_field_2d(self):
        from anaspike.dataclasses.bins import calculate_bin_sums_2d
        from anaspike.dataclasses.field import Field2D
        field = Field2D(self.coords,
                        np.array([[1, 2],
                                  [3, 4],
                                  [5, 6],
                                  [7, 8]]))
        expected_sums = np.array([[[1, 2], [10, 12]],
                                  [[0, 0], [0, 0]],
                                  [[0, 0], [5, 6]]
                                 ])
        bin_sums = calculate_bin_sums_2d(self.bins, field)
        np.testing.assert_array_almost_equal(bin_sums.elements, expected_sums)

class TestCalculateBinMeans(TestClassSetup):
    def test_coords_outside_bins(self):
        from anaspike.dataclasses.bins import calculate_bin_means_2d
        from anaspike.dataclasses.field import Field2D
        field = Field2D(self.coords_outside_bins,
                        np.array(np.arange(4) + 1, dtype=np.float64))
        with self.assertRaises(ValueError):
            calculate_bin_means_2d(self.bins, field)

    def test_calculate_bin_means_2d_sclar_field(self):
        from anaspike.dataclasses.bins import calculate_bin_means_2d
        from anaspike.dataclasses.field import Field2D
        field = Field2D(self.coords,
                        np.array(np.arange(4) + 1, dtype=np.float64))
        expected_means = np.array([[1 / 1, 6 / 2],
                                   [np.nan, np.nan],
                                   [np.nan, 3 / 1]])
        bin_means = calculate_bin_means_2d(self.bins, field)
        np.testing.assert_array_almost_equal(bin_means.elements, expected_means)

    def test_calculate_bin_means_2d_vector_field_2d(self):
        from anaspike.dataclasses.bins import calculate_bin_means_2d
        from anaspike.dataclasses.field import Field2D
        field = Field2D(self.coords,
                        np.array([[1, 2],
                                  [3, 4],
                                  [5, 6],
                                  [7, 8]]))
        expected_means = np.array([[[1 / 1, 2 / 1],  [10 / 2, 12 / 2]],
                                   [[np.nan, np.nan], [np.nan, np.nan]],
                                   [[np.nan, np.nan], [5 / 1, 6 / 1]]])
        bin_means = calculate_bin_means_2d(self.bins, field)
        np.testing.assert_array_almost_equal(bin_means.elements, expected_means)

    def test_calculate_bin_means_2d_another_scalar_field(self):
        import numpy as np
        from anaspike.dataclasses.bins import ContigBins2D
        from anaspike.dataclasses.grid import RegularGrid2D, RegularGrid1D
        from anaspike.dataclasses.interval import Interval
        from anaspike.dataclasses.field import Field2D
        from anaspike.dataclasses.coords2d import Coords2D
        from anaspike.dataclasses.bins import calculate_bin_means_2d
        field = Field2D(
                Coords2D(x=np.array([0.1, 0.4, 0.6, 0.8, 0.2, 0.3, 0.5]),
                         y=np.array([1.5, 0.7, 0.2, 0.9, 1.9, 1.4, 0.8])),
                np.array([1, 2, 3, 4, 5, 6, 7]))
        bin_grid = RegularGrid2D(RegularGrid1D.from_interval_given_n(Interval(0., 1.),
                                                                     n=3,
                                                                     endpoint=True),
                                 RegularGrid1D.from_interval_given_n(Interval(0., 2.),
                                                                     n=4,
                                                                     endpoint=True))
        bins = ContigBins2D[RegularGrid2D].with_median_labels(bin_grid)
        expected_firing_rates = np.array([[np.nan, 3.],
                                          [2.    , 11./2],
                                          [12./3 , np.nan]]).T
                                               
        binned_field = calculate_bin_means_2d(bins, field)
        np.testing.assert_array_almost_equal(binned_field.elements, expected_firing_rates)

