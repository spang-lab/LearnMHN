import unittest
import warnings
import numpy as np
import mhn.full_state_space.fisher
from mhn.full_state_space import Likelihood, ModelConstruction
from mhn.training.likelihood_cmhn import gradient_and_score
from mhn.training.state_containers import StateContainer


class TestFisher(unittest.TestCase):

    def test_cython_fisher(self):
        """Test the Fisher information matrix for cMHN.
        """
        n = 4
        log_theta = ModelConstruction.random_theta(n)

        p_th = Likelihood.generate_pTh(log_theta)
        FIM = np.zeros((n * n, n * n))
        X = [[l, k, j, i] for i in range(2) for j in range(2)
             for k in range(2) for l in range(2)]

        for i, x in enumerate(X):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grad = gradient_and_score(
                    log_theta,
                    StateContainer(
                        np.array(x, dtype=np.int32).reshape(1, n)
                    ))[0].flatten()
            FIM += p_th[i] * np.outer(grad, grad)

        for use_cuda in [True, False]:
            with self.subTest(use_cuda):
                if use_cuda:
                    if mhn.cuda_available() != \
                            mhn.CUDA_AVAILABLE:
                        self.skipTest("CUDA not available for testing")
                np.testing.assert_allclose(
                    FIM,
                    mhn.full_state_space.fisher.fisher(
                        log_theta=log_theta, omhn=False, use_cuda=use_cuda)
                )

    def test_omhn_fisher_theta_theta(self):
        """Test upper left Block of oMHN Fisher information matrix."""

        n = 6
        o_log_theta = 2 * np.random.random((n + 1, n)) - 1
        c_log_theta = mhn.model.oMHN(
            o_log_theta).get_equivalent_classical_mhn().log_theta

        np.testing.assert_allclose(
            mhn.full_state_space.fisher.cython_fisher(c_log_theta),
            mhn.full_state_space.fisher.omhn_fisher(o_log_theta)[:n**2, :n**2]
        )

    def test_omhn_fisher_omega_theta(self):
        """Test off-diagonal Blocks of oMHN Fisher information matrix."""

        n = 6

        log_theta = 2 * np.random.random((n + 1, n)) - 1
        fisher_matrix = mhn.full_state_space.fisher.omhn_fisher(log_theta)

        np.testing.assert_allclose(
            fisher_matrix[n**2:, :n**2],
            fisher_matrix[:n**2, n**2:].T)

        fisher_omega_theta = fisher_matrix[n**2:, :n**2]

        for j in range(n):
            for s in range(n):
                for t in range(n):
                    np.testing.assert_allclose(
                        fisher_omega_theta[j, s * n + t],
                        -sum(fisher_matrix[i * n + j, s * n + t]
                             for i in range(n) if i != j)
                    )

    def test_omhn_fisher_omega_omega(self):
        """Test lower right Block of oMHN Fisher information matrix."""

        n = 6
        log_theta = 2 * np.random.random((n + 1, n)) - 1
        fisher_matrix = mhn.full_state_space.fisher.omhn_fisher(log_theta)

        fisher_omega_omega = fisher_matrix[n**2:, n**2:]

        np.testing.assert_allclose(
            fisher_omega_omega,
            fisher_omega_omega.T
        )

        for j in range(n):
            for t in range(n):
                np.testing.assert_allclose(
                    fisher_omega_omega[j, t],
                    sum(
                        sum(fisher_matrix[i * n + j, s * n + t]
                            for s in range(n) if s != t)
                        for i in range(n) if i != j
                    ))


if __name__ == "__main__":
    unittest.main()
