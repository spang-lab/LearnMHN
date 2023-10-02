"""
This submodule contains functions that can be used to compute the scores and gradients for the OmegaMHN
using state-space restriction.
"""
# author(s): Stefan Vocht

import numpy as np
from typing import Callable

from ..ssr import state_space_restriction
from ..ssr.state_containers import StateContainer


def L1(theta: np.ndarray, eps: float = 1e-05) -> float:
    """
    Computes the L1 penalty
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return np.sum(np.sqrt(theta_**2 + eps))


def L1_(theta: np.ndarray, eps: float = 1e-05) -> np.ndarray:
    """
    Derivative of the L1 penalty
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return theta_ / np.sqrt(theta_**2 + eps)


def build_regularized_score_func(gradient_and_score_function: Callable):
    """
    This function gets a function which can compute a gradient and a score at the same time and returns a function
    which computes the score and adds a L1 regularization
    """
    def reg_score_func(theta: np.ndarray, states: StateContainer, lam: float,
                                          n: int, score_grad_container: list) -> float:
        """
        Computes the score using state space restriction with L1 regularization

        :param theta: current theta
        :param states: states observed in the data
        :param lam: regularization parameter
        :param n: size of theta (nxn)
        :param score_grad_container: a list that enables this function to communicate with the gradient function
        :return: regularized score
        """
        theta = theta.reshape((n+1, n))
        grad, score = gradient_and_score_function(theta, states)
        score_grad_container[0] = grad

        return -(score - lam * L1(theta))
    return reg_score_func


def build_regularized_gradient_func(gradient_and_score_function: Callable):
    """
    This function gets a function which can compute a gradient and a score at the same time and returns a function
    which computes the gradient and adds the gradient of the L1 regularization
    """
    def reg_gradient_func(theta: np.ndarray, states: StateContainer, lam: float,
                                             n: int, score_grad_container: list) -> np.ndarray:
        """
        Computes the gradient state space restriction with L1 regularization

        :param theta: current theta
        :param states: states observed in the data
        :param lam: regularization parameter
        :param n: size of theta (nxn)
        :param score_grad_container: a list that enables this function to communicate with the score function
        :return: regularized gradient
        """
        theta_ = theta.reshape((n+1, n))
        grad = score_grad_container[0]
        if grad is None:
            grad, score = gradient_and_score_function(theta_, states)
        return -(grad - lam * L1_(theta_)).flatten()
    return reg_gradient_func


def gradient_and_score(omega_theta: np.ndarray, mutation_data: StateContainer):
    """
    Computes the score as well as the gradient in log format.

    :param omega_theta: theta matrix for the OmegaMHN (shape: (n+1) x n), last row contains observation rates
    :param mutation_data: StateContainer object containing the data which is used for training
    :returns: tuple containing the gradient and the score of the current OmegaMHN
    """
    n = omega_theta.shape[1]
    if omega_theta.shape[0] != n+1:
        raise ValueError("omega_theta must be a 2d numpy array with n columns and n+1 rows")
    # subtract observation rates from each element in each column
    equivalent_vanilla_mhn = omega_theta[:-1] - omega_theta[-1]
    # undo changes to the diagonal
    equivalent_vanilla_mhn[range(n), range(n)] += omega_theta[-1]
    # compute the score and gradient on this theta matrix
    grad, score = state_space_restriction.gradient_and_score(equivalent_vanilla_mhn, mutation_data)
    # compute the gradient for the observation rates
    observation_rates_gradient = -(np.sum(grad, axis=0) - grad.diagonal())
    omega_gradient = np.empty((n+1, n), dtype=np.double)
    omega_gradient[:n] = grad
    omega_gradient[-1] = observation_rates_gradient

    return omega_gradient, score


def test():
    import mhn
    _n = 10
    _theta = mhn.original.ModelConstruction.random_theta(_n)
    _omega_theta = np.zeros((_n+1, _n))
    _omega_theta[:_n] = _theta
    _random_data = np.random.choice([0, 1], (20, _n), p=(0.5, 0.5))
    _mutation_data = StateContainer(_random_data)
    print(gradient_and_score(_omega_theta, _mutation_data))
    return gradient_and_score(_omega_theta, _mutation_data)
