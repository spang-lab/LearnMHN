"""
Unit tests for the mhn.mcmc module.
"""

import sys
import unittest
if sys.version_info < (3, 10):
    raise unittest.SkipTest("mhn.mcmc requires Python 3.9+")

import numpy as np
import mhn
from mhn.training.state_containers import StateContainer
from mhn.mcmc.kernels import RWMKernel, MALAKernel, smMALAKernel
from mhn.mcmc.mcmc import MCMC
from mhn.optimizers import Penalty
import scipy.stats

np.random.seed(0)
data = np.random.randint(2, size=(100,3), dtype=np.int32)
data_container = StateContainer(data)
data_size = data.shape[0]
optimizer = mhn.optimizers.oMHNOptimizer().load_data_matrix(data)
np.random.seed(0)
lam = optimizer.lambda_from_cv()
model = optimizer.train(lam=lam)
shape = (4, 3)
size = 12
lam = 0.001


def grad_and_log_likelihood(theta):
    grad, lik = mhn.training.likelihood_omhn.gradient_and_score(
        theta.reshape(shape), data_container)
    return data_size * grad, data_size * lik


def log_l2_prior(theta):
    return -lam * np.sum(theta**2)


def log_l2_prior_grad(theta):
    return -2 * lam * theta.reshape(shape)


def log_l2_prior_hessian(theta):
    return -2 * lam * np.eye(size)


def log_l1_prior(theta):
    return -lam * np.sum(np.abs(theta))


def log_l1_prior_grad(theta):
    return -lam * np.sign(theta.reshape(shape))


def log_l1_prior_hessian(theta):
    return np.zeros((size, size))


initial_theta = np.random.normal(loc=0, scale=1 / np.sqrt(2 * lam), size=size)


class TestKernels(unittest.TestCase):
    """Tests for MCMC kernel classes."""

    def test_rwm_kernel_init(self):
        """Test RWMKernel initialization."""
        kernel = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
        )
        self.assertEqual(kernel.step_size, 0.1)
        self.assertEqual(kernel.shape, shape)
        self.assertEqual(kernel.size, size)

    def test_rwm_kernel_get_params(self):
        """Test RWMKernel get_params method."""
        kernel = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
        )

        result = kernel.get_params(initial_theta)

        assert (np.allclose(result.log_likelihood,
                grad_and_log_likelihood(initial_theta)[1]))
        assert (np.allclose(result.log_prior, log_l2_prior(initial_theta)))

    def test_rwm_kernel_propose(self):
        """Test RWMKernel propose method."""
        kernel = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res = kernel.get_params(initial_theta)
        kernel.propose(
            initial_theta, prev_step_res)

    def test_rwm_kernel_log_accept(self):
        """Test RWMKernel log_accept method."""
        kernel = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
        )

        step1 = initial_theta
        res1 = kernel.get_params(step1)
        step2 = initial_theta + 0.1
        res2 = kernel.get_params(step2)

        accept_ratio = kernel.log_accept(step1, res1, step2, res2)

        assert (np.allclose(
            accept_ratio,
            log_l2_prior(step2)
            - log_l2_prior(step1)
            + grad_and_log_likelihood(step2)[1]
            - grad_and_log_likelihood(step1)[1]))

    def test_rwm_kernel_one_step(self):
        """Test RWMKernel one_step method."""
        kernel = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res = kernel.get_params(initial_theta)
        new_step, new_step_res, acceptance_ratio, accepted = kernel.one_step(
            initial_theta, prev_step_res, return_info=True
        )

        # Check output types and shapes
        self.assertEqual(new_step.shape, initial_theta.shape)
        self.assertIsInstance(acceptance_ratio, float)
        self.assertIn(accepted, [0, 1])

    def test_rwm_kernel_seed_reproducibility(self):
        """Test RWMKernel reproducibility with same seed."""
        # First run with seed 42
        kernel1 = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.one_step(initial_theta, prev_step_res1)

        # Second run with same seed 42
        kernel2 = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.one_step(initial_theta, prev_step_res2)

        # Results should be identical
        np.testing.assert_array_equal(new_step1, new_step2)

    def test_rwm_kernel_different_seeds_differ(self):
        """Test that RWMKernel with different seeds produces different results."""
        # First run with seed 42
        kernel1 = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.propose(initial_theta, prev_step_res1)

        # Second run with different seed 99
        kernel2 = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(99))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.propose(initial_theta, prev_step_res2)

        # Results should be different (with high probability)
        self.assertFalse(np.allclose(new_step1, new_step2))

    def test_mala_kernel_init(self):
        """Test MALAKernel initialization."""
        kernel = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
        )
        self.assertEqual(kernel.step_size, 0.1)
        self.assertEqual(kernel.shape, shape)

    def test_mala_kernel_get_params(self):
        """Test MALAKernel get_params method."""
        kernel = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
        )

        result = kernel.get_params(initial_theta)

        assert (np.allclose(result.log_likelihood,
                grad_and_log_likelihood(initial_theta)[1]))
        assert (np.allclose(result.log_prior, log_l2_prior(initial_theta)))
        self.assertTrue(hasattr(result, "mu"))

    def test_mala_kernel_propose(self):
        """Test MALAKernel propose method."""
        kernel = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res = kernel.get_params(initial_theta)
        kernel.propose(
            initial_theta, prev_step_res)

    def test_mala_kernel_log_accept(self):
        """Test MALAKernel log_accept method."""
        kernel = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
        )

        step1 = initial_theta
        res1 = kernel.get_params(step1)
        step2 = initial_theta + 0.1
        res2 = kernel.get_params(step2)

        accept_ratio = kernel.log_accept(step1, res1, step2, res2)

        assert (np.allclose(
            accept_ratio,
            log_l2_prior(step2)
            - log_l2_prior(step1)
            + grad_and_log_likelihood(step2)[1]
            - grad_and_log_likelihood(step1)[1]
            + scipy.stats.multivariate_normal.logpdf(
                step1,
                mean=step2 + kernel.step_size / 2 *
                (grad_and_log_likelihood(step2)[
                 0] + log_l2_prior_grad(step2)).flatten(),
                cov=kernel.step_size * np.eye(size))
            - scipy.stats.multivariate_normal.logpdf(
                step2,
                mean=step1 + kernel.step_size / 2 *
                (grad_and_log_likelihood(step1)[
                 0] + log_l2_prior_grad(step1)).flatten(),
                cov=kernel.step_size * np.eye(size))
        ))

    def test_mala_kernel_one_step(self):
        """Test MALAKernel one_step method."""
        kernel = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res = kernel.get_params(initial_theta)
        new_step, new_step_res, acceptance_ratio, accepted = kernel.one_step(
            initial_theta, prev_step_res, return_info=True
        )

        # Check output types and shapes
        self.assertEqual(new_step.shape, initial_theta.shape)
        self.assertIsInstance(acceptance_ratio, (float, np.floating))
        self.assertIn(accepted, [0, 1])

    def test_mala_kernel_seed_reproducibility(self):
        """Test MALAKernel reproducibility with same seed."""
        # First run with seed 123
        kernel1 = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(123))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.one_step(initial_theta, prev_step_res1)

        # Second run with same seed 123
        kernel2 = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(123))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.one_step(initial_theta, prev_step_res2)

        # Results should be identical
        np.testing.assert_array_equal(new_step1, new_step2)

    def test_mala_kernel_different_seeds_differ(self):
        """Test that MALAKernel with different seeds produces different results."""
        # First run with seed 123
        kernel1 = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(123))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.propose(initial_theta, prev_step_res1)

        # Second run with different seed 321
        kernel2 = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(321))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.propose(initial_theta, prev_step_res2)

        # Results should be different (with high probability)
        self.assertFalse(np.allclose(new_step1, new_step2))

    def test_smmala_kernel_init(self):
        """Test smMALAKernel initialization."""
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.1,
            shape=shape,
        )
        self.assertEqual(kernel.step_size, 0.1)
        self.assertEqual(kernel.shape, shape)
        self.assertFalse(kernel.use_cuda)

    def test_smmala_kernel_get_params(self):
        """Test smMALAKernel get_params method."""
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.1,
            shape=shape,
        )

        result = kernel.get_params(initial_theta)

        assert (np.allclose(result.log_likelihood,
                grad_and_log_likelihood(initial_theta)[1]))
        assert (np.allclose(result.log_prior, log_l2_prior(initial_theta)))
        assert (np.allclose(result.gradient, grad_and_log_likelihood(
            initial_theta)[0] + log_l2_prior_grad(initial_theta)))
        assert (np.allclose(result.G, -log_l2_prior_hessian(initial_theta) +
                mhn.full_state_space.fisher.fisher(initial_theta.reshape(shape))))
        assert (np.allclose(result.cholesky, np.linalg.cholesky(result.G)))
        assert hasattr(result, 'mu')
        assert (np.allclose(result.det_sqrt, np.sqrt(np.linalg.det(result.G))))

    def test_smmala_kernel_get_params_cuda(self):
        """Test smMALAKernel get_params method with CUDA."""
        if mhn.cuda_available() != mhn.CUDA_AVAILABLE:
            self.skipTest("CUDA not available, skipping CUDA test.")
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.1,
            shape=shape,
        )
        cuda_kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.1,
            shape=shape,
            use_cuda=True
        )

        result = kernel.get_params(initial_theta)
        result_cuda = cuda_kernel.get_params(initial_theta)

        # Check that results are close (accounting for possible floating point differences)
        self.assertTrue(np.allclose(result.log_likelihood,
                        result_cuda.log_likelihood))
        self.assertTrue(np.allclose(result.log_prior, result_cuda.log_prior))
        self.assertTrue(np.allclose(result.gradient, result_cuda.gradient))
        self.assertTrue(np.allclose(result.G, result_cuda.G))
        self.assertTrue(np.allclose(result.cholesky, result_cuda.cholesky))
        self.assertTrue(np.allclose(result.mu, result_cuda.mu))
        self.assertTrue(np.allclose(result.det_sqrt, result_cuda.det_sqrt))

    def test_linalg_error_non_positive_definite(self):
        """Test that smMALAKernel raises LinAlgError for non-positive-definite metric."""

        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l1_prior,
            log_prior_grad=log_l1_prior_grad,
            log_prior_hessian=log_l1_prior_hessian,
            step_size=0.1,
            shape=shape,
        )

        with self.assertRaises(np.linalg.LinAlgError):
            kernel.get_params(initial_theta)

    def test_smmala_kernel_propose(self):
        """Test smMALAKernel propose method."""
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.01,  # Smaller step size for stability
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res = kernel.get_params(initial_theta)
        kernel.propose(
            initial_theta, prev_step_res)

    def test_smmala_kernel_log_accept(self):
        """Test smMALAKernel log_accept method."""
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.01,
            shape=shape,
        )

        step1 = initial_theta
        res1 = kernel.get_params(step1)
        step2 = initial_theta + 0.001
        res2 = kernel.get_params(step2)

        accept_ratio = kernel.log_accept(step1, res1, step2, res2)

        assert (np.allclose(
            accept_ratio,
            log_l2_prior(step2)
            - log_l2_prior(step1)
            + grad_and_log_likelihood(step2)[1]
            - grad_and_log_likelihood(step1)[1]
            + scipy.stats.multivariate_normal.logpdf(
                step1,
                mean=step2 + kernel.step_size / 2 * np.linalg.inv(res2.G) @
                (grad_and_log_likelihood(step2)[
                 0] + log_l2_prior_grad(step2)).flatten(),
                cov=kernel.step_size * np.linalg.inv(res2.G))
            - scipy.stats.multivariate_normal.logpdf(
                step2,
                mean=step1 + kernel.step_size / 2 * np.linalg.inv(res1.G) @
                (grad_and_log_likelihood(step1)[
                 0] + log_l2_prior_grad(step1)).flatten(),
                cov=kernel.step_size * np.linalg.inv(res1.G))
        ))

    def test_smmala_kernel_one_step(self):
        """Test smMALAKernel one_step method."""
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.001,  # Very small step size for stability
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )
        prev_step_res = kernel.get_params(initial_theta)
        new_step, new_step_res, acceptance_ratio, accepted = kernel.one_step(
            initial_theta, prev_step_res, return_info=True
        )

        # Check output types and shapes
        self.assertEqual(new_step.shape, initial_theta.shape)
        self.assertIsInstance(acceptance_ratio, (float, np.floating))
        self.assertIn(accepted, [0, 1])

    def test_smmala_kernel_seed_reproducibility(self):
        """Test smMALAKernel reproducibility with same seed."""
        # First run with seed 456
        kernel1 = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.001,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(456))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.one_step(initial_theta, prev_step_res1)

        # Second run with same seed 456
        kernel2 = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.001,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(456))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.one_step(initial_theta, prev_step_res2)

        # Results should be identical
        np.testing.assert_array_equal(new_step1, new_step2)

    def test_smmala_kernel_different_seeds_differ(self):
        """Test that smMALAKernel with different seeds produces different results."""
        # First run with seed 456
        kernel1 = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.001,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(456))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.propose(initial_theta, prev_step_res1)

        # Second run with different seed 654
        kernel2 = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.001,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(654))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.propose(initial_theta, prev_step_res2)

        # Results should be different (with high probability)
        self.assertFalse(np.allclose(new_step1, new_step2))


class TestMCMC(unittest.TestCase):
    """Integration tests for the MCMC sampler."""

    def test_init_from_optimizer(self):
        sampler = MCMC(optimizer=optimizer, n_chains=2)
        self.assertEqual(sampler.n_chains, 2)

    def test_init_from_model_data_penalty(self):
        """Initialize using model, data and penalty from optimizer."""
        sampler = MCMC(
            mhn_model=model,
            data=data,
            penalty=Penalty.L2,
            n_chains=2
        )
        self.assertEqual(sampler.n_chains, 2)

    def test_init_invalid_combinations(self):
        """Check errors when providing conflicting or incomplete arguments."""
        params = {
            "optimizer": optimizer,
            "data": data,
            "mhn_model": model,
            "penalty": Penalty.L2,
            "log_prior": (log_l2_prior, log_l2_prior_grad, log_l2_prior_hessian),
        }

        for args in [
            ("data",),
            ("mhn_model",),
            ("penalty",),
            ("log_prior",),
            ("data", "mhn_model"),
            ("data", "penalty"),
            ("data", "log_prior"),
            ("mhn_model", "penalty"),
            ("mhn_model", "log_prior"),
            ("penalty", "log_prior"),
            ("data", "mhn_model", "penalty", "log_prior"),

        ]:
            with self.assertRaises(ValueError):
                MCMC(**{k: params[k] for k in args})

        with self.assertRaises(ValueError):
            optimizer2 = mhn.optimizers.oMHNOptimizer()
            optimizer2.results = optimizer.result
            optimizer2._data = optimizer._data
            MCMC(optimizer=optimizer2)
        with self.assertWarns(UserWarning):
            MCMC(optimizer=optimizer, penalty=Penalty.L2)
        with self.assertWarns(UserWarning):
            MCMC(optimizer=optimizer, log_prior=(
                log_l2_prior, log_l2_prior_grad, log_l2_prior_hessian))
        with self.assertRaises(ValueError):
            MCMC(optimizer=optimizer, kernel_class=smMALAKernel)
        with self.assertRaises(ValueError):
            MCMC(mhn_model=model, log_prior=(log_l2_prior, log_l2_prior_grad),
                 kernel_class=smMALAKernel)
        with self.assertRaises(ValueError):
            MCMC(mhn_model=model, log_prior=(log_l2_prior),
                 kernel_class=MALAKernel)

    def test_run_rwm_basic(self):
        sampler = MCMC(
            optimizer=optimizer,
            kernel_class=RWMKernel,
            step_size=0.1,
            n_chains=2,
            thin=1,
            seed=42
        )
        result = sampler.run(stopping_crit=None, max_steps=10, verbose=False)
        self.assertEqual(result.shape, (2, 10, size))

        sampler = MCMC(
            mhn_model=model,
            data=data,
            penalty=Penalty.L2,
            kernel_class=RWMKernel,
            step_size=0.1,
            n_chains=2,
            thin=1,
            seed=42
        )
        result = sampler.run(stopping_crit=None, max_steps=10, verbose=False)
        self.assertEqual(result.shape, (2, 10, size))

    def test_run_mala_basic(self):
        sampler = MCMC(
            optimizer=optimizer,
            kernel_class=MALAKernel,
            step_size=0.1,
            n_chains=2,
            thin=1,
            seed=42
        )
        result = sampler.run(stopping_crit=None, max_steps=10, verbose=False)
        self.assertEqual(result.shape, (2, 10, size))

    def test_run_smmala_basic(self):
        sampler = MCMC(
            mhn_model=model,
            data=data,
            log_prior=(log_l2_prior, log_l2_prior_grad, log_l2_prior_hessian),
            kernel_class=smMALAKernel,
            step_size=0.001,
            n_chains=2,
            thin=1,
            seed=42
        )
        sampler.initial_step = np.random.normal(
            loc=0, scale=1 / np.sqrt(2 * lam), size=(2, 1, size))
        result = sampler.run(stopping_crit=None,
                             max_steps=3, verbose=False)
        self.assertEqual(result.shape, (2, 3, size))

    def test_seed_reproducibility(self):
        sam1 = MCMC(
            optimizer=optimizer,
            step_size=0.1,
            seed=123,
            n_chains=2,
            thin=1
        )
        sam1.run(stopping_crit=None, max_steps=5, verbose=False)
        sam2 = MCMC(
            optimizer=optimizer,
            step_size=0.1,
            seed=123,
            n_chains=2,
            thin=1
        )
        sam2.run(stopping_crit=None, max_steps=5, verbose=False)
        np.testing.assert_array_equal(sam1.log_thetas, sam2.log_thetas)

    def test_rhat_and_ess(self):
        sampler = MCMC(
            optimizer=optimizer,
            step_size=0.1,
            seed=42,
            n_chains=3,
            thin=1
        )
        sampler.run(stopping_crit=None, max_steps=20, verbose=False)
        r = sampler.rhat()
        e = sampler.ess()
        self.assertEqual(len(r), sampler.size)
        self.assertEqual(len(e), sampler.size)
        self.assertTrue(np.all(r > 0.99))
        self.assertTrue(np.all(e > 0))

    def test_acceptance_rate(self):
        sampler = MCMC(
            optimizer=optimizer,
            step_size=0.1,
            seed=42,
            n_chains=3,
            thin=1
        )
        sampler.run(stopping_crit=None, max_steps=20, verbose=False)
        rates = sampler.acceptance()
        self.assertEqual(len(rates), 3)
        self.assertTrue(np.all((rates >= 0) & (rates <= 1)))

    def test_tune_stepsize(self):
        sampler = MCMC(
            optimizer=optimizer,
            step_size="auto",
            n_chains=2,
            seed=42,
            thin=1
        )
        val = sampler.tune_stepsize(n_steps=5, verbose=False, tol=0.3)
        self.assertIsInstance(val, float)
        self.assertEqual(sampler.step_size, val)


if __name__ == "__main__":
    unittest.main()
