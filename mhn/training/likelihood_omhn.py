"""
This submodule contains functions that can be used to compute the scores and gradients for the oMHN.
"""
# author(s): Stefan Vocht

from __future__ import annotations
from typing import Callable

import numpy as np

from . import likelihood_cmhn
from .state_containers import StateContainer


def _internal_gradient_and_score(omega_theta: np.ndarray, mutation_data: StateContainer, grad_and_score_func: Callable) -> tuple[np.ndarray, float]:
    """
    Computes the log-likelihood score and its gradient using the specified gradient and score function.

    Args:
        omega_theta (np.ndarray): Theta matrix for the oMHN (shape: (n+1) x n), where the last row contains observation rates.
        mutation_data (StateContainer): Data used for training.
        grad_and_score_func (Callable): Function that computes the gradient and score.

    Returns:
        tuple[np.ndarray, float]: The gradient (as a numpy array) and the score (as a float).
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


def gradient_and_score(omega_theta: np.ndarray, mutation_data: StateContainer) -> tuple[np.ndarray, float]:
    """
    Computes the log-likelihood score and gradient using both CPU and GPU.

    This function uses the CPU implementation for data points with few mutations and CUDA (if available) for data points with many mutations.
    If CUDA is not installed, the function defaults to CPU.

    Args:
        omega_theta (np.ndarray): Theta matrix for the oMHN (shape: (n+1) x n), where the last row contains observation rates.
        mutation_data (StateContainer): Data used for training.

    Returns:
        tuple[np.ndarray, float]: The gradient (as a numpy array) and the score (as a float).
    """
    return _internal_gradient_and_score(omega_theta, mutation_data, likelihood_cmhn.gradient_and_score)


def cpu_gradient_and_score(omega_theta: np.ndarray, mutation_data: StateContainer) -> tuple[np.ndarray, float]:
    """
    Computes the log-likelihood score and gradient on the CPU.

    Args:
        omega_theta (np.ndarray): Theta matrix for the oMHN (shape: (n+1) x n), where the last row contains observation rates.
        mutation_data (StateContainer): Data used for training.

    Returns:
        tuple[np.ndarray, float]: The gradient (as a numpy array) and the score (as a float).
    """
    return _internal_gradient_and_score(omega_theta, mutation_data, likelihood_cmhn.cpu_gradient_and_score)


def cpu_score(omega_theta: np.ndarray, mutation_data: StateContainer) -> float:
    """
    Computes the log-likelihood score on the CPU.

    Args:
        omega_theta (np.ndarray): Theta matrix for the oMHN (shape: (n+1) x n), where the last row contains observation rates.
        mutation_data (StateContainer): Data used for training.

    Returns:
        float: Log-likelihood score of the given oMHN for the provided data.
    """
    n = omega_theta.shape[1]
    if omega_theta.shape[0] != n+1:
        raise ValueError("omega_theta must be a 2d numpy array with n columns and n+1 rows")
    # subtract observation rates from each element in each column
    equivalent_vanilla_mhn = omega_theta[:-1] - omega_theta[-1]
    # undo changes to the diagonal
    equivalent_vanilla_mhn[range(n), range(n)] += omega_theta[-1]
    score = likelihood_cmhn.cpu_score(equivalent_vanilla_mhn, mutation_data)
    return score


def cuda_gradient_and_score(omega_theta: np.ndarray, mutation_data: StateContainer):
    """
    Computes the log-likelihood score and gradient on the GPU.

    **Note:** This function can only be used if the mhn package was compiled with CUDA.

    Args:
        omega_theta (np.ndarray): Theta matrix for the oMHN (shape: (n+1) x n), where the last row contains observation rates.
        mutation_data (StateContainer): Data used for training.

    Returns:
        tuple[np.ndarray, float]: The gradient (as a numpy array) and the score (as a float).
    """
    return _internal_gradient_and_score(omega_theta, mutation_data, likelihood_cmhn.cuda_gradient_and_score)

