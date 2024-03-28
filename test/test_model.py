import unittest

import numpy as np

from mhn import model
from mhn.original import ModelConstruction, UtilityFunctions, Likelihood


class TestMHN(unittest.TestCase):
    """
    Tests methods of the MHN class.
    """

    def setUp(self) -> None:
        """
        Preparation for each test
        """
        np.random.seed(0)  # set random seed for reproducibility

    def test_sample_artificial_data(self):
        """
        Tests if the distribution sampled by sample_artificial_data() equals the distribution represented by the MHN.
        """
        theta = ModelConstruction.random_theta(8)
        mhn_object = model.MHN(theta)
        p_th = Likelihood.generate_pTh(theta)

        art_data = mhn_object.sample_artificial_data(500_000)
        p_data = UtilityFunctions.data_to_pD(art_data)
        np.testing.assert_allclose(p_th, p_data, atol=1e-3)

    def test_sample_trajectories(self):
        """
        Tests if the distribution sampled by sample_trajectories() equals the distribution represented by the MHN.
        """
        n = 8
        theta = ModelConstruction.random_theta(n)
        mhn_object = model.MHN(theta)
        p_th = Likelihood.generate_pTh(theta)

        trajectories, obs_times = mhn_object.sample_trajectories(500_000, [])
        cross_sec_data = list(map(
            lambda trajectory: [1 if i in trajectory else 0 for i in range(n)],
            trajectories
        ))
        cross_sec_data = np.array(cross_sec_data)
        p_data = UtilityFunctions.data_to_pD(cross_sec_data)
        np.testing.assert_allclose(p_th, p_data, atol=1e-3)

    def test_sample_trajectories_initial_state(self):
        """
        Tests if the initial state parameter works correctly.
        """
        n = 8
        theta = ModelConstruction.random_theta(n)
        mhn_object = model.MHN(theta)
        mhn_object.events = ["A" * i for i in range(n)]

        initial_event_num = 2
        initial_bin_state = np.zeros(n, dtype=np.int32)
        initial_bin_state[:initial_event_num] = 1
        initial_event_state = ["A" * i for i in range(initial_event_num)]

        np.random.seed(0)
        trajectories_1, obs_times_1 = mhn_object.sample_trajectories(100, initial_state=initial_bin_state)
        np.random.seed(0)
        trajectories_2, obs_times_2 = mhn_object.sample_trajectories(100, initial_state=initial_event_state)

        np.testing.assert_array_equal(obs_times_1, obs_times_2)
        self.assertListEqual(trajectories_1, trajectories_2)

    def test_compute_marginal_likelihood(self):
        """
        Tests if the probabilities yielded by compute_marginal_likelihood() match the actual probability distribution.
        """
        n = 5
        theta = ModelConstruction.random_theta(n)
        mhn_object = model.MHN(theta)

        p_th = Likelihood.generate_pTh(theta)

        # code from https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
        d = np.arange(2**n)
        all_possible_states = ((d[:, None] & (1 << np.arange(n))) > 0).astype(np.int32)

        for i in range(2**n):
            p = mhn_object.compute_marginal_likelihood(all_possible_states[i])
            self.assertAlmostEqual(p, p_th[i], 10)


class TestOmegaMHN(unittest.TestCase):
    """
    Tests methods of the OmegaMHN class.
    """

    def setUp(self) -> None:
        """
        Preparation for each test
        """
        np.random.seed(0)  # set random seed for reproducibility

    def test_sample_trajectories(self):
        """
        Tests if the distribution sampled by sample_trajectories() equals the distribution represented by the oMHN.
        """
        n = 8
        theta = ModelConstruction.random_theta(n)
        theta = np.vstack((theta, np.random.random(n)))
        mhn_object = model.OmegaMHN(theta)
        p_th = Likelihood.generate_pTh(mhn_object.get_equivalent_classical_mhn().log_theta)

        trajectories, obs_times = mhn_object.sample_trajectories(500_000, [])
        cross_sec_data = list(map(
            lambda trajectory: [1 if i in trajectory else 0 for i in range(n)],
            trajectories
        ))
        cross_sec_data = np.array(cross_sec_data)
        p_data = UtilityFunctions.data_to_pD(cross_sec_data)
        np.testing.assert_allclose(p_th, p_data, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
