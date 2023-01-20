import unittest

from mhn.original import ModelConstruction
from mhn.ssr import matrix_exponential, state_storage

import numpy as np
import scipy


class TestMatrixExponential(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(0)

    def test_correctness(self):
        n = 6
        theta = ModelConstruction.random_theta(n)
        q = ModelConstruction.build_q(theta)
        all_mutated_data = np.ones((1, n), dtype=np.int32)
        b = np.random.random(2**n)

        for t in np.arange(0, 6, 0.6):
            container = state_storage.StateAgeStorage(all_mutated_data, np.array([t]))
            result1 = scipy.linalg.expm(t * q).dot(b)
            result2 = matrix_exponential.restricted_expm(theta, b, container, 1e-8)
            np.testing.assert_array_equal(np.around(result1, decimals=3), np.around(result2, decimals=3))


if __name__ == '__main__':
    unittest.main()
