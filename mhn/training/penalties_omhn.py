"""
This module contains penalties used during training for regularization.
"""
# author(s): Stefan Vocht

import numpy as np
from typing import Callable

from .state_containers import StateContainer


def l1(theta: np.ndarray, eps: float = 1e-05) -> float:
    """
    Computes the L1 penalty. Base rates are not penalized.

    Args:
        theta (np.ndarray): Input array representing model parameters.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-05.

    Returns:
        float: Computed L1 penalty.
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return np.sum(np.sqrt(theta_ ** 2 + eps))


def l1_(theta: np.ndarray, eps: float = 1e-05) -> np.ndarray:
    """
    Computes the derivative of the L1 penalty.

    Args:
        theta (np.ndarray): Input array representing model parameters.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-05.

    Returns:
        np.ndarray: Gradient of the L1 penalty with respect to theta.
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return theta_ / np.sqrt(theta_ ** 2 + eps)


def l2(theta: np.ndarray) -> float:
    """
    Computes the L2 penalty. Base rates are not penalized.

    Args:
        theta (np.ndarray): Input array representing model parameters.

    Returns:
        float: Computed L2 penalty.
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return np.sum(theta_ ** 2)


def l2_(theta: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the L2 penalty.

    Args:
        theta (np.ndarray): Input array representing model parameters.

    Returns:
        np.ndarray: Gradient of the L2 penalty with respect to theta.
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return 2. * theta_


def sym_sparse(omega_theta: np.ndarray, eps: float = 1e-05) -> float:
    """
    Computes a penalty that induces sparsity and soft symmetry.

    Args:
        omega_theta (np.ndarray): Input array representing model parameters.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-05.

    Returns:
        float: Computed penalty.
    """
    theta_copy = omega_theta.copy()
    np.fill_diagonal(theta_copy, 0)
    theta_without_omega = theta_copy[:-1]
    n = theta_without_omega.shape[0]
    theta_sum = np.sum(
        np.sqrt(
            theta_without_omega.T ** 2 + theta_without_omega ** 2 - theta_without_omega.T * theta_without_omega + eps)
    )
    # remove all eps that were added to the diagonal (which should be zero) in the equation above
    theta_sum -= n * np.sqrt(eps)
    # due to the symmetry of the formula, we get twice the value of what we want, so halve it
    theta_sum *= 0.5
    omega_sum = np.sum(np.sqrt(theta_copy[-1] ** 2 + eps))
    return theta_sum + omega_sum


def sym_sparse_deriv(omega_theta: np.ndarray, eps: float = 1e-05) -> np.ndarray:
    """
    Computes the derivative of the sparsity and symmetry penalty.

    Args:
        omega_theta (np.ndarray): Input array representing parameters.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-05.

    Returns:
        np.ndarray: Gradient of the sym_sparse penalty.
    """
    theta_copy = omega_theta.copy()
    np.fill_diagonal(theta_copy, 0)
    theta_without_omega = theta_copy[:-1]
    theta_sum_denominator = 2 * np.sqrt(
        theta_without_omega.T ** 2 + theta_without_omega ** 2 - theta_without_omega.T * theta_without_omega + eps
    )
    theta_sum_numerator = 2 * theta_without_omega - theta_without_omega.T
    theta_derivative = theta_sum_numerator / theta_sum_denominator
    omega_derivative = theta_copy[-1] / np.sqrt(theta_copy[-1] ** 2 + eps)
    return np.vstack((theta_derivative, omega_derivative))


def build_regularized_score_func(gradient_and_score_function: Callable, penalty_function: Callable = l1) -> Callable:
    """
    Wraps a gradient-and-score function to include regularization.

    Args:
        gradient_and_score_function (Callable): Function that computes gradient and score.
        penalty_function (Callable, optional): Regularization penalty function. Defaults to l1.

    Returns:
        Callable: Regularized score function.
    """

    def reg_score_func(theta: np.ndarray, states: StateContainer, lam: float,
                       n: int, score_grad_container: list) -> float:
        """
        Computes a regularized score using state space restriction.

        Args:
            theta (np.ndarray): Current model parameters.
            states (StateContainer): Observed states from data.
            lam (float): Regularization parameter.
            n (int): Size of the theta (n+1xn).
            score_grad_container (list): Container for communication between score and gradient functions.

        Returns:
            float: Regularized score.
        """
        theta = theta.reshape((n + 1, n))
        grad, score = gradient_and_score_function(theta, states)
        score_grad_container[0] = grad

        return -(score - lam * penalty_function(theta))

    return reg_score_func


def build_regularized_gradient_func(gradient_and_score_function: Callable, penalty_derivative: Callable = l1_) -> Callable:
    """
    Wraps a gradient-and-score function to include regularization gradient.

    Args:
        gradient_and_score_function (Callable): Function that computes gradient and score.
        penalty_derivative (Callable, optional): Regularization penalty derivative function. Defaults to l1_.

    Returns:
        Callable: Regularized gradient function.
    """

    def reg_gradient_func(theta: np.ndarray, states: StateContainer, lam: float,
                          n: int, score_grad_container: list) -> np.ndarray:
        """
        Computes a regularized gradient using state space restriction.

        Args:
            theta (np.ndarray): Current model parameters.
            states (StateContainer): Observed states from data.
            lam (float): Regularization parameter.
            n (int): Size of the theta (n+1xn).
            score_grad_container (list): Container for communication between score and gradient functions.

        Returns:
            np.ndarray: Regularized gradient.
        """
        theta_ = theta.reshape((n + 1, n))
        grad = score_grad_container[0]
        if grad is None:
            grad, score = gradient_and_score_function(theta_, states)
        return -(grad - lam * penalty_derivative(theta_)).flatten()

    return reg_gradient_func
