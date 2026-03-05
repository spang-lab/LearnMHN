"""
This submodule contains the Random-Walk Metropolis, Metropolis-Adjusted
Langevin Algorithm (MALA) and simplified manifold MALA kernels and a
base kernel base class. Those are responsible for the actual steps of
MCMC.
"""
# author(s): Y. Linda Hu

from __future__ import annotations

import numpy as np
import scipy.linalg
from typing import Callable
from ..full_state_space.fisher import fisher as get_fisher
from typing import NamedTuple


class Kernel:
    """Base class for kernels used in MCMC sampling.

    Attributes:
        grad_and_log_likelihood (tuple[Callable[[np.ndarray], tuple[np.ndarray, float]]]):
            function that takes a flattened (!) log_theta matrix and
            returns a tuple of the gradient of the log-likelihood and
            the log-likelihood itself.

            **Important**: Here, the likelihood should *not* be
            normalized by the number of samples like
            in Schill et al. (2019).
        log_prior: Callable[[np.ndarray], float]: function that takes a
            flattened (!) log_theta matrix and returns the log-prior.
            This corresponds to the negative penalty term times lambda,
            multiplied by the number of samples, if applicable.
        shape: tuple[int, int]: the shape of the log_theta matrix, i.e.
            (n_events + 1, n_events) for oMHN and (n_events, n_events)
            for cMHN.
        rng: np.random.Generator | None = None: random number generator
            to be used for sampling. If None, a new RNG will be created.
    """

    def __init__(
        self,
        grad_and_log_likelihood: tuple[Callable[[np.ndarray], tuple[np.ndarray, float]]],
        log_prior: Callable[[np.ndarray], float],
        shape: tuple[int, int],
        rng: np.random.Generator | None = None,
    ):
        self.grad_and_log_likelihood = grad_and_log_likelihood
        self.log_prior = log_prior
        self.rng = rng or np.random.Generator(np.random.PCG64())
        self.shape = shape
        self.size = shape[0]*shape[1]


class smMALAResult(NamedTuple):
    """Results for one smMALAKernel step.

    Args:
        log_likelihood (float): log-likelihood of the current step
        log_prior (float): log-prior of the current step
        gradient (np.ndarray): gradient of the log-posterior at the
            current step
        G (np.ndarray): metric tensor at the current step
        cholesky (np.ndarray): Cholesky decomposition of the metric
            tensor at the current step
        mu (np.ndarray): mean of the proposal distribution at the
            current step
        det_sqrt (float): square root of the determinant of the metric
            tensor at the current step
    """
    log_likelihood: float
    log_prior: float
    gradient: np.ndarray
    G: np.ndarray
    cholesky: np.ndarray
    mu: np.ndarray
    det_sqrt: float


class smMALAKernel(Kernel):
    """Class for kernel used in simplified manifold MALA. Extends
    Kernel.

    Attributes:

        step_size (float): step size for the smMALA sampling. This is
            the :math:`\\epsilon^2/4` in the proposal distribution.
        grad_and_log_likelihood (tuple[Callable[[np.ndarray], tuple[np.ndarray, float]]]):
            function that takes a flattened (!) log_theta matrix and
            returns a tuple of the gradient of the log-likelihood and
            the log-likelihood itself.

            **Important**: Here, the likelihood should *not* be
            normalized by the number of samples like
            in Schill et al. (2019)
        log_prior: Callable[[np.ndarray], float]: function that takes a
            flattened (!) log_theta matrix and returns the log-prior.
            This corresponds to the negative penalty term times lambda,
            multiplied by the number of samples, if applicable.
        log_prior_grad: Callable[[np.ndarray], np.ndarray]: function
            that takes a flattened (!) log_theta matrix and returns the
            gradient of the log-prior.
        log_prior_hessian: Callable[[np.ndarray], np.ndarray]: function
            that takes a flattened (!) log_theta matrix and returns the
            Hessian of the log-prior.
        shape: tuple[int, int]: the shape of the log_theta matrix, i.e.
            (n_events + 1, n_events) for oMHN and (n_events, n_events)
            for cMHN.
        use_cuda: bool = False: whether to use CUDA for the computation
            of the Fisher information matrix. Only applicable if
            mhn.cuda_available() says that CUDA is available.
        rng: np.random.Generator | None = None: random number generator
            to be used for sampling. If None, a new RNG will be created.

    """

    def __init__(
        self,
        step_size: float,
        grad_and_log_likelihood: tuple[Callable[[np.ndarray], tuple[np.ndarray, float]]],
        log_prior: Callable[[np.ndarray], float],
        log_prior_grad: Callable[[np.ndarray], np.ndarray],
        log_prior_hessian: Callable[[np.ndarray], np.ndarray],
        shape: tuple[int, int],
        use_cuda: bool = False,
        rng: np.random.Generator | None = None,
    ):

        super().__init__(grad_and_log_likelihood=grad_and_log_likelihood,
                         log_prior=log_prior, shape=shape, rng=rng)

        self.step_size = step_size
        self.log_prior_grad = log_prior_grad
        self.log_prior_hessian = log_prior_hessian
        self.use_cuda = use_cuda

    def propose(self, prev_step: np.ndarray, prev_step_res: smMALAResult
                ) -> tuple[np.ndarray, smMALAResult]:
        """Propose a new step based on the previous step and its
        results.
        This is done according to

        :math:`\\mathcal N\\bigg(\\theta + \\frac{\\epsilon}{2} G(\\theta)^{-1} \\frac{\\partial \\log p(\\theta)}{\\partial \\theta}, \\epsilon G(\\theta)^{-1}\\bigg)`
        
        where
        
        :math:`G(\\theta) = I(\\theta) - H(\\log \\pi(\\theta))`
        
        with :math:`I` the Fisher information matrix and :math:`H` the Hessian of the log-prior.


        Args:
            prev_step (np.array): the previous step
            prev_step_res (smMALAResult): Results of the previous step.

        Returns:
            tuple[np.ndarray, smMALAResult]: the new step and its
            results.
        """

        # draw random normal number
        z = self.rng.normal(size=prev_step.size)

        # transform with inverse transformed cholesky matrix
        y = np.sqrt(self.step_size) * scipy.linalg.solve_triangular(
            prev_step_res.cholesky.T, z, lower=False
        )

        new_step = prev_step_res.mu + y
        return new_step, self.get_params(new_step)

    def log_accept(
            self,
            prev_step: np.ndarray,
            prev_step_res: smMALAResult,
            new_step: np.ndarray,
            new_step_res: smMALAResult) -> float:
        """Give acceptance ratio of accepting the new step.

        This is given by

        :math:`\\frac{p(\\theta' | D) q(\\theta | \\theta')\\pi(\\theta')}{p(\\theta|D) q(\\theta' | \\theta) \\pi(\\theta)}`
        
        where :math:`p(.| D)` is the likelihood, :math:`\\pi(.)` is the
        prior and :math:`q(.|.)` is the proposal distribution.

        Args:
            prev_step (np.ndarray): the previous step
            prev_step_res (smMALAResult): Results of the previous step.
            new_step (np.ndarray): the new step
            new_step_res (smMALAResult): Results of the new step.

        Returns:
            float: the acceptance ratio
        """

        # p(theta' | D) q(theta | theta') pr(theta') / p(theta|D) q(theta' | theta) pr(theta)

        acceptance_ratio = (
            new_step_res.log_likelihood
            + new_step_res.log_prior
            + self.log_q(
                theta_proposed=prev_step,
                G=new_step_res.G,
                det_sqrt_G=new_step_res.det_sqrt,
                mu=new_step_res.mu,
            )
            - prev_step_res.log_likelihood
            - prev_step_res.log_prior
            - self.log_q(
                theta_proposed=new_step,
                G=prev_step_res.G,
                det_sqrt_G=prev_step_res.det_sqrt,
                mu=prev_step_res.mu,
            )
        )

        return acceptance_ratio

    def one_step(
        self,
        prev_step: np.ndarray,
        prev_step_res: smMALAResult,
        return_info: bool = False
    ) -> tuple[np.ndarray, smMALAResult] | tuple[np.ndarray, smMALAResult, float, int]:
        """Perform one smMALA step.

        Args:
            prev_step (np.ndarray): the previous step
            prev_step_res (smMALAResult): Results of the previous
                step.
            return_info (bool, optional): Whether to return the
                acceptance ratio and acceptance indicator. Defaults to
                False.

        Returns:
            tuple[np.ndarray, smMALAResult] |
            tuple[np.ndarray, smMALAResult, float, int]: The new step
                and its results, and optionally acceptance ratio and
                acceptance indicator.
        """

        new_step, new_step_res = self.propose(prev_step, prev_step_res)
        acceptance_ratio = self.log_accept(
            prev_step, prev_step_res, new_step, new_step_res
        )

        if np.log(self.rng.random()) < acceptance_ratio:
            if return_info:
                return (new_step, new_step_res, acceptance_ratio, 1)
            return (new_step, new_step_res)
        else:
            if return_info:
                return (prev_step, prev_step_res, acceptance_ratio, 0)
            return (prev_step, prev_step_res)

    def get_params(self, initial_step: np.ndarray) -> smMALAResult:
        """Get the results for a given step.

        Args:
            initial_step (np.ndarray): The step for which to get the
                results.

        Returns:
            smMALAResult: The results for the given step.
        """

        # Get gradient, likelihood and G matrix for new theta
        log_likelihood_grad, log_likelihood = self.grad_and_log_likelihood(
            initial_step)
        log_prior = self.log_prior(initial_step)

        log_posterior_grad = log_likelihood_grad + \
            self.log_prior_grad(initial_step)

        fisher = get_fisher(
            log_theta=initial_step.reshape(self.shape),
            omhn=self.shape[0] == self.shape[1] + 1,
            use_cuda=self.use_cuda,
        )
        G = fisher - self.log_prior_hessian(initial_step)
        try:
            cholesky = scipy.linalg.cholesky(G, lower=True)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                "The cholesky decomposition of the metric tensor" \
                "failed. This likely means that the metric tensor is " \
                "not positive definite. Check that the hessian of " \
                "the log-prior is negative definite and that the " \
                "and, if necessary, decrease the stepsize."
            ) from e
        det_sqrt = np.diag(cholesky).prod()

        # Get mu, the mean of the proposal distribution w.r.t. the new theta
        # this is log_theta + 0.5 * STEP_SIZE * G^-1 * gradient
        y = scipy.linalg.solve_triangular(
            cholesky.T,
            scipy.linalg.solve_triangular(
                cholesky,
                log_posterior_grad.flatten(),
                lower=True,
            ),
            lower=False,
        )
        mu = initial_step.flatten() + 0.5 * self.step_size * y

        return smMALAResult(
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            gradient=log_posterior_grad,
            G=G,
            cholesky=cholesky,
            mu=mu,
            det_sqrt=det_sqrt,
        )

    def log_q(
        self,
        theta_proposed: np.ndarray,
        G: np.ndarray,
        det_sqrt_G: float,
        mu: np.ndarray,
    ) -> float:
        """Compute the logarithm of the proposal distribution density
            q(theta_new | theta) for the smMALA algorithm.
            This is according to https://en.wikipedia.org/wiki/Multivariate_normal_distribution
            #Density_function.

        Args:
            theta_proposed (np.ndarray): New proposed theta
            G (np.ndarray): Metric tensor w.r.t. the old theta
            det_sqrt_G (float): Square root of the determinant of the
                metric tensor w.r.t. the old theta
            mu (np.ndarray): Mean of the proposal distribution w.r.t.
                the old theta

        Returns:
            float: The logarithm of the proposal distribution
                q(theta_new | theta) density
        """
        # we can leave out the constant factor
        # (2 * np.pi) ** (n_events**2 )
        # in the denominator, as well as the scaling
        # STEP_SIZE ** (n_events**2)
        # 1/sqrt(det(G^-1)) = sqrt(det(G))
        return -0.5 * (theta_proposed - mu).T @ G @ (
            theta_proposed - mu
        ) / self.step_size + np.log(det_sqrt_G)


class MALAResult(NamedTuple):
    """Results for one smMALAKernel step.

    Args:
        log_likelihood (float): log-likelihood of the current step
        log_prior (float): log-prior of the current step
        mu (np.ndarray): mean of the proposal distribution at the
            current step
    """
    log_likelihood: float
    log_prior: float
    mu: np.ndarray


class MALAKernel(Kernel):
    """Class for kernel used in the Metropolis-Adjusted Langevin
    Algorithm. Extends Kernel.

    Attributes:
        step_size (float): step size for the MALA sampling.
        grad_and_log_likelihood (tuple[Callable[[np.ndarray], tuple[np.ndarray, float]]]):
            function that takes a flattened (!) log_theta matrix and
            returns a tuple of the gradient of the log-likelihood and
            the log-likelihood itself.

            **Important**: Here, the likelihood should *not* be
            normalized by the number of samples like
            in Schill et al. (2019)
        log_prior: Callable[[np.ndarray], float]: function that takes a
            flattened (!) log_theta matrix and returns the log-prior.
            This corresponds to the negative penalty term times lambda,
            multiplied by the number of samples, if applicable.
        log_prior_grad: Callable[[np.ndarray], np.ndarray]: function
            that takes a flattened (!) log_theta matrix and returns the
            gradient of the log-prior.
        shape: tuple[int, int]: the shape of the log_theta matrix, i.e.
            (n_events + 1, n_events) for oMHN and (n_events, n_events)
            for cMHN.
        rng: np.random.Generator | None = None: random number generator
            to be used for sampling. If None, a new RNG will be created.

    """

    def __init__(
        self,
        step_size: float,
        grad_and_log_likelihood: tuple[Callable[[np.ndarray], tuple[np.ndarray, float]]],
        log_prior: Callable[[np.ndarray], float],
        log_prior_grad: Callable[[np.ndarray], np.ndarray],
        shape: tuple[int, int],
        rng: np.random.Generator | None = None,
    ):

        super().__init__(grad_and_log_likelihood=grad_and_log_likelihood,
                         log_prior=log_prior, shape=shape, rng=rng)

        self.step_size = step_size
        self.log_prior_grad = log_prior_grad

    def propose(self, prev_step: np.ndarray, prev_step_res: MALAResult
                ) -> tuple[np.ndarray, MALAResult]:
        """Propose a new step based on the previous step and its 
        results.

        Args:
            prev_step (np.array): the previous step
            prev_step_res (MALAResult): Results of the previous step.

        Returns:
            tuple[np.ndarray, MALAResult]: the new step and its
            results.
        """
        z = self.rng.normal(size=self.size)
        new_step = prev_step_res.mu + np.sqrt(self.step_size) * z

        return new_step, self.get_params(new_step)

    def log_accept(
            self,
            prev_step: np.ndarray,
            prev_step_res: MALAResult,
            new_step: np.ndarray,
            new_step_res: MALAResult) -> float:
        """Give acceptance ratio of accepting the new step.

        Args:
            prev_step (np.ndarray): the previous step
            prev_step_res (MALAResult): Results of the previous step.
            new_step (np.ndarray): the new step
            new_step_res (MALAResult): Results of the new step.

        Returns:
            float: the acceptance ratio
        """
        # p(theta' | D) q(theta | theta') pr(theta') /
        # p(theta|D) q(theta' | theta) pr(theta)

        acceptance_ratio = (
            new_step_res.log_likelihood
            + new_step_res.log_prior
            + self.log_q(
                theta_proposed=prev_step,
                mu=new_step_res.mu,
            )
            - prev_step_res.log_likelihood
            - prev_step_res.log_prior
            - self.log_q(
                theta_proposed=new_step,
                mu=prev_step_res.mu,
            )
        )

        return acceptance_ratio

    def one_step(
        self,
        prev_step: np.ndarray,
        prev_step_res: MALAResult,
        return_info: bool = False
    ) -> tuple[np.ndarray, MALAResult] | tuple[np.ndarray, MALAResult, float, int]:
        """Perform one MALA step.

        Args:
            prev_step (np.ndarray): the previous step
            prev_step_res (MALAResult): Results of the previous step.
            return_info (bool, optional): Whether to return the
                acceptance ratio and acceptance indicator. Defaults to
                False.

        Returns:
            tuple[np.ndarray, MALAResult] |
            tuple[np.ndarray, MALAResult, float, int]: The new step and
                its results, and optionally acceptance ratio and
                acceptance indicator.
        """
        new_step, new_step_res = self.propose(prev_step, prev_step_res)
        acceptance_ratio = self.log_accept(
            prev_step, prev_step_res, new_step, new_step_res
        )

        if np.log(self.rng.random()) < acceptance_ratio:
            if return_info:
                return (new_step, new_step_res, acceptance_ratio, 1)
            return (new_step, new_step_res)
        else:
            if return_info:
                return (prev_step, prev_step_res, acceptance_ratio, 0)
            return (prev_step, prev_step_res)

    def get_params(self, initial_step: np.ndarray) -> MALAResult:
        """Get the results for a given step.

        Args:
            initial_step (np.ndarray): The step for which to get the
                results.

        Returns:
            MALAResult: The results for the given step.
        """

        log_likelihood_grad, log_likelihood = self.grad_and_log_likelihood(
            initial_step)
        log_posterior_grad = log_likelihood_grad + \
            self.log_prior_grad(initial_step.reshape(self.shape))

        mu = initial_step.flatten() \
            + 0.5 * self.step_size * log_posterior_grad.flatten()
        return MALAResult(
            log_likelihood=log_likelihood,
            log_prior=self.log_prior(initial_step.reshape(self.shape)),
            mu=mu,
        )

    def log_q(
        self,
        theta_proposed: np.ndarray,
        mu: np.ndarray,
    ) -> float:
        """Compute the logarithm of the proposal distribution density
        q(theta_new | theta) for the MMALA algorithm.
        This is according to https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function.

        Args:
            theta_new (np.ndarray): New proposed theta
            G (np.ndarray): Metric tensor w.r.t. the old theta
            det_sqrt_G (float): Square root of the determinant of the
            metric tensor w.r.t. the old theta
            mu (np.ndarray): Mean of the proposal distribution w.r.t.
            the old theta

        Returns:
            float: The logarithm of the proposal distribution 
            q(theta_new | theta) density
        """
        # we can leave out the constant factor (2 * np.pi) ** (n_events**2 )
        # in the denominator, as well as the scaling STEP_SIZE ** (n_events**2)
        return -0.5 * np.sum((theta_proposed - mu) ** 2) / self.step_size


class RWMResult(NamedTuple):
    """Results for one RWMKernel step.

    Args:
        log_likelihood (float): log-likelihood of the current step
        log_prior (float): log-prior of the current step
    """
    log_likelihood: float
    log_prior: float


class RWMKernel(Kernel):
    """Class for kernel used in simplified Random-Walk Metropolis.
    Extends Kernel.

    Attributes:
        grad_and_log_likelihood (tuple[Callable[[np.ndarray], tuple[np.ndarray, float]]]):
            function that takes a flattened (!) log_theta matrix and
            returns a tuple of the gradient of the log-likelihood and
            the log-likelihood itself.

            **Important**: Here, the likelihood should *not* be
            normalized by the number of samples like
            in Schill et al. (2019)
        log_prior: Callable[[np.ndarray], float]: function that takes a
            flattened (!) log_theta matrix and returns the log-prior.
            This corresponds to the negative penalty term times lambda,
            multiplied by the number of samples, if applicable.
        scale (float): scale of the normal proposal distribution.
        shape: tuple[int, int]: the shape of the log_theta matrix, i.e.
            (n_events + 1, n_events) for oMHN and (n_events, n_events)
            for cMHN.
        rng: np.random.Generator | None = None: random number generator
            to be used for sampling. If None, a new RNG will be created.

    """

    def __init__(
        self,
        grad_and_log_likelihood: tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], float]],
        log_prior: Callable[[np.ndarray], float],
        step_size: float,
        shape: tuple[int, int],
        rng: np.random.Generator | None = None,
    ):
        super().__init__(grad_and_log_likelihood=grad_and_log_likelihood,
                         log_prior=log_prior, shape=shape, rng=rng)

        self.step_size = step_size
        self.sigma = step_size * np.eye(self.size)

    def propose(self, prev_step: np.ndarray, prev_step_res: RWMResult):
        """Propose a new step based on the previous step and its
        results.
        The proposal is made according to a normal distribution centered
        at the previous step with standard deviation `step_size`.

        Args:
            prev_step (np.array): the previous step
            prev_step_res (RWMResult): Results of the previous step.

        Returns:
            tuple[np.ndarray, RWMResult]: the new step and its
            results.
        """
        # draw random normal number
        new_step = self.rng.normal(
            loc=prev_step,
            scale=self.step_size,
            size=self.size,
        )

        return new_step, self.get_params(new_step)

    def log_accept(
            self,
            prev_step: np.ndarray,
            prev_step_res: RWMResult,
            new_step: np.ndarray,
            new_step_res: RWMResult) -> float:
        """Give acceptance ratio of accepting the new step. This is
        given by
        p(new_step | D) pr(new_step) / p(prev_step|D) pr(prev_step)
        where p(.| D) is the likelihood and pr(.) is the prior.

        Args:
            prev_step (np.ndarray): the previous step
            prev_step_res (RWMResult): Results of the previous step.
            new_step (np.ndarray): the new step
            new_step_res (RWMResult): Results of the new step.

        Returns:
            float: the acceptance ratio
        """
        # p(theta' | D) pr(theta') / p(theta|D) pr(theta)

        acceptance_ratio = (
            new_step_res.log_likelihood
            + new_step_res.log_prior
            - prev_step_res.log_likelihood
            - prev_step_res.log_prior
        )

        return acceptance_ratio

    def one_step(
            self,
            prev_step: np.ndarray,
            prev_step_res: RWMResult,
            return_info: bool = False):
        """Perform one RWM step.

        Args:
            prev_step (np.ndarray): the previous step
            prev_step_res (RWMResult): Results of the previous
                step.
            return_info (bool, optional): Whether to return the
                acceptance ratio and acceptance indicator. Defaults to
                False.

        Returns:
            tuple[np.ndarray, RWMResult] |
            tuple[np.ndarray, RWMResult, float, int]: The new step
            and its results, and optionally acceptance ratio and
            acceptance indicator.
        """

        new_step, new_step_res = self.propose(prev_step, prev_step_res)
        acceptance_ratio = self.log_accept(
            prev_step, prev_step_res, new_step, new_step_res
        )

        if np.log(self.rng.random()) < acceptance_ratio:
            if return_info:
                return (new_step, new_step_res, acceptance_ratio, 1)
            return (new_step, new_step_res)
        else:
            if return_info:
                return (prev_step, prev_step_res, acceptance_ratio, 0)
            return (prev_step, prev_step_res)

    def get_params(self, initial_step):
        """Get the results for a given step.

        Args:
            initial_step (np.ndarray): The step for which to get the
                results.

        Returns:
            RWMResult: The results for the given step.
        """
        return RWMResult(
            self.grad_and_log_likelihood(initial_step.reshape(self.shape))[1],
            self.log_prior(initial_step.reshape(self.shape))
        )
