"""
This module contains penalties used during training for regularization.
"""
# author(s): Stefan Vocht


from typing import Callable

import numpy as np

from .state_containers import StateContainer


def l1(theta: np.ndarray, eps: float = 1e-05) -> float:
    """
    Computes the L1 penalty. Base rates are not penalized.

    Args:
        theta (np.ndarray): The input array.
        eps (float): Small value that avoids division by zero

    Returns:
        float: The L1 penalty value.
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return np.sum(np.sqrt(theta_**2 + eps))


def l1_(theta: np.ndarray, eps: float = 1e-05) -> np.ndarray:
    """
    Derivative of the L1 penalty.

    Args:
        theta (np.ndarray): The input array.
        eps (float): Small value that avoids division by zero

    Returns:
        np.ndarray: The gradient of the L1 penalty with respect to theta.
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return theta_ / np.sqrt(theta_**2 + eps)


def l2(theta: np.ndarray) -> float:
    """
    Computes the L2 penalty. Base rates are not penalized.

    Args:
        theta (np.ndarray): The input array.

    Returns:
        float: The L2 penalty value.
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return np.sum(theta_ ** 2)


def l2_(theta: np.ndarray) -> np.ndarray:
    """
    Derivative of the L2 penalty.

    Args:
        theta (np.ndarray): The input array.

    Returns:
        np.ndarray: The gradient of the L2 penalty with respect to theta.
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return 2. * theta_


def sym_sparse(theta: np.ndarray, eps: float = 1e-05) -> float:
    """
    A penalty which induces sparsity and soft symmetry.

    Args:
        theta (np.ndarray): The input array.
        eps (float): Small value that avoids division by zero

    Returns:
        float: The penalty value.
    """
    theta_copy = theta.copy()
    np.fill_diagonal(theta_copy, 0)
    n = theta_copy.shape[0]
    theta_sum = np.sum(
        np.sqrt(theta_copy.T**2 + theta_copy**2 - theta_copy.T * theta_copy + eps)
    )
    # remove all eps that were added to the diagonal (which should be zero) in the equation above
    theta_sum -= n * np.sqrt(eps)
    # due to the symmetry of the formula, we get twice the value of what we want, so halve it
    theta_sum *= 0.5
    return theta_sum


def sym_sparse_deriv(theta: np.ndarray, eps: float = 1e-05) -> np.ndarray:
    """
    Derivative of the sym_sparse penalty.

    Args:
        theta (np.ndarray): The input array.
        eps (float): Small value that avoids division by zero

    Returns:
        np.ndarray: The gradient of the sym_sparse penalty with respect to theta.
    """
    theta_copy = theta.copy()
    np.fill_diagonal(theta_copy, 0)
    theta_sum_denominator = 2 * np.sqrt(
        theta_copy.T**2 + theta_copy**2 - theta_copy.T * theta_copy + eps
    )
    theta_sum_numerator = 2 * theta_copy - theta_copy.T
    theta_derivative = theta_sum_numerator / theta_sum_denominator
    return theta_derivative


def build_regularized_score_func(gradient_and_score_function: Callable, penalty_function: Callable = l1):
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
        theta = theta.reshape((n, n))
        grad, score = gradient_and_score_function(theta, states)
        score_grad_container[0] = grad

        return -(score - lam * penalty_function(theta))
    return reg_score_func


def build_regularized_gradient_func(gradient_and_score_function: Callable, penalty_derivative: Callable = l1_):
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
        n = n or int(np.sqrt(theta.size))
        theta_ = theta.reshape((n, n))
        grad = score_grad_container[0]
        if grad is None:
            grad, score = gradient_and_score_function(theta_, states)
        return -(grad - lam * penalty_derivative(theta_)).flatten()
    return reg_gradient_func

