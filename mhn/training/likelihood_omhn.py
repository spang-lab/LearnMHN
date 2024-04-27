"""
This submodule contains functions that can be used to compute the scores and gradients for the OmegaMHN
using state-space restriction.
"""
# author(s): Stefan Vocht

import numpy as np
from typing import Callable

from . import state_space_restriction
from .state_containers import StateContainer


def _internal_gradient_and_score(omega_theta: np.ndarray, mutation_data: StateContainer, grad_and_score_func: Callable):
    """
    Computes the log-likelihood score as well as its gradient using the given gradient and score function.

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
    grad, score = grad_and_score_func(equivalent_vanilla_mhn, mutation_data)
    # compute the gradient for the observation rates
    observation_rates_gradient = -(np.sum(grad, axis=0) - grad.diagonal())
    omega_gradient = np.empty((n+1, n), dtype=np.double)
    omega_gradient[:n] = grad
    omega_gradient[-1] = observation_rates_gradient

    return omega_gradient, score


def gradient_and_score(omega_theta: np.ndarray, mutation_data: StateContainer):
    """
    Computes the log-likelihood score as well as its gradient.

    This function computes the gradient using both the CPU AND CUDA (only if CUDA is installed).
    It will compute the gradients for data points with few mutations using the CPU implementation
    and compute the gradients for data points with many mutations using CUDA.
    If CUDA is not installed on your device, this function will only use the CPU implementation.

    :param omega_theta: theta matrix for the OmegaMHN (shape: (n+1) x n), last row contains observation rates
    :param mutation_data: StateContainer object containing the data which is used for training
    :returns: tuple containing the gradient and the score of the current OmegaMHN
    """
    return _internal_gradient_and_score(omega_theta, mutation_data, state_space_restriction.gradient_and_score)


def cpu_gradient_and_score(omega_theta: np.ndarray, mutation_data: StateContainer):
    """
    Computes the log-likelihood score as well as its gradient on the CPU.

    :param omega_theta: theta matrix for the OmegaMHN (shape: (n+1) x n), last row contains observation rates
    :param mutation_data: StateContainer object containing the data which is used for training
    :returns: tuple containing the gradient and the score of the current OmegaMHN
    """
    return _internal_gradient_and_score(omega_theta, mutation_data, state_space_restriction.cpu_gradient_and_score)


def cpu_score(omega_theta: np.ndarray, mutation_data: StateContainer):
    """
    Computes the log-likelihood score on the CPU.

    Args:
        omega_theta: np.ndarray
            theta matrix for the oMHN (shape: (n+1) x n), last row contains observation rates
        mutation_data: StateContainer
            StateContainer object containing the data which is used for training

    Returns:
        log-likelihood score of the given oMHN for the given data.
    """
    n = omega_theta.shape[1]
    if omega_theta.shape[0] != n+1:
        raise ValueError("omega_theta must be a 2d numpy array with n columns and n+1 rows")
    # subtract observation rates from each element in each column
    equivalent_vanilla_mhn = omega_theta[:-1] - omega_theta[-1]
    # undo changes to the diagonal
    equivalent_vanilla_mhn[range(n), range(n)] += omega_theta[-1]
    score = state_space_restriction.cpu_score(equivalent_vanilla_mhn, mutation_data)
    return score


def cuda_gradient_and_score(omega_theta: np.ndarray, mutation_data: StateContainer):
    """
    Computes the log-likelihood score as well as the gradient on the GPU.

    **This function can only be used if the mhn package was compiled with CUDA.**

    :param omega_theta: theta matrix for the OmegaMHN (shape: (n+1) x n), last row contains observation rates
    :param mutation_data: StateContainer object containing the data which is used for training
    :returns: tuple containing the gradient and the score of the current OmegaMHN
    """
    return _internal_gradient_and_score(omega_theta, mutation_data, state_space_restriction.cuda_gradient_and_score)

