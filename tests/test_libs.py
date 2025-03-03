import unittest

import numpy as np



class TestCorrcoeff(unittest.TestCase):
    def test_corrcoef_scipy_vs_numpy(self):
        from scipy.stats import pearsonr
        rng = np.random.default_rng(seed=42)
        n_vars = 3
        n_samples = 100
        arr = rng.random((n_vars, n_samples))
        np_corrcoef = np.corrcoef(arr)
        scipy_corrcoef = np.array([[pearsonr(r1, r2).statistic for r1 in arr] for r2 in arr])
        np.testing.assert_allclose(np_corrcoef, scipy_corrcoef)



if __name__ == '__main__':
    unittest.main(verbosity=2)
