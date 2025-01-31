"""
This submodule implements RegularizedOptimization.R in Python.

It contains functions to learn an cMHN on the *full state-space* for a given data distribution and implements the L1 regularization.
"""
# author(s): Stefan Vocht

import numpy as np
from scipy.optimize import minimize

from . import Likelihood
from . import ModelConstruction

from typing import Callable


def L1(theta: np.ndarray, eps: float = 1e-05) -> float:
    """
    Computes the L1 penalty for the given theta matrix.

    Args:
        theta (np.ndarray): The theta matrix representing the cMHN.
        eps (float, optional): A small epsilon value, mainly used for the derivative. Default is 1e-05.

    Returns:
        float: The L1 penalty for the given theta matrix.
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return np.sum(np.sqrt(theta_**2 + eps))


def L1_(theta: np.ndarray, eps: float = 1e-05) -> np.ndarray:
    """
    Computes the derivative of the L1 penalty for the given theta matrix.

    Args:
        theta (np.ndarray): The theta matrix representing the cMHN.
        eps (float, optional): A small epsilon value to prevent division by zero. Default is 1e-05.

    Returns:
        np.ndarray: The derivative of the L1 penalty.
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return theta_ / np.sqrt(theta_**2 + eps)


def score_reg(theta: np.ndarray, pD: np.ndarray, lam: float, n: int = None, pth_space: np.ndarray = None) -> float:
    """
    Computes the score with L1 regularization for the given theta matrix.

    Args:
        theta (np.ndarray): The theta matrix representing the cMHN.
        pD (np.ndarray): Distribution provided by the training data.
        lam (float): Tuning parameter lambda for regularization.
        n (int, optional): The number of columns/rows of the theta matrix. Default is None.
        pth_space (np.ndarray, optional): Optional parameter used for communication with the gradient function to improve performance.

    Returns:
        float: The score of the current cMHN, penalized with L1 regularization.
    """
    n = n or int(np.sqrt(theta.size))
    theta = theta.reshape((n, n))

    return -(Likelihood.score(theta, pD, pth_space) - lam * L1(theta))


def grad_reg(theta: np.ndarray, pD: np.ndarray, lam: float, n: int = 0, pth_space: np.ndarray = None) -> np.ndarray:
    """
    Computes the gradient with L1 regularization for the given theta matrix.

    Args:
        theta (np.ndarray): The theta matrix representing the cMHN.
        pD (np.ndarray): Distribution provided by the training data.
        lam (float): Tuning parameter lambda for regularization.
        n (int, optional): The number of columns/rows of the theta matrix. Default is 0.
        pth_space (np.ndarray, optional): Optional parameter used to avoid recalculating pth, improving performance.

    Returns:
        np.ndarray: The gradient of the L1-regularized score.
    """
    n = n or int(np.sqrt(theta.size))
    theta_ = theta.reshape((n, n))

    return -(Likelihood.grad(theta_, pD, pth_space) - lam * L1_(theta_)).flatten()


def learn_MHN(pD: np.ndarray, init: np.ndarray = None, lam: float = 0, maxit: int = 5000,
              trace: bool = False, reltol: float = 1e-07, round_result: bool = True,
              callback: Callable = None, score_func: Callable = score_reg, jacobi: Callable = grad_reg) -> np.ndarray:
    """
    Trains a cMHN to fit a given probability distribution.

    Args:
        pD (np.ndarray): Probability distribution used to train the new model.
        init (np.ndarray, optional): Starting point for the training (initial theta). Default is None.
        lam (float, optional): Tuning parameter lambda for regularization. Default is 0.
        maxit (int, optional): Maximum number of training iterations. Default is 5000.
        trace (bool, optional): If True, convergence messages are printed (see scipy.optimize.minimize). Default is False.
        reltol (float, optional): Gradient norm must be less than reltol before successful termination. Default is 1e-07.
        round_result (bool, optional): If True, the result is rounded to two decimal places. Default is True.
        callback (Callable, optional): Function called after each iteration, must take theta as an argument. Default is None.
        score_func (Callable, optional): Score function used for training. Default is score_reg.
        jacobi (Callable, optional): Gradient function used for training. Default is grad_reg.

    Returns:
        np.ndarray: The trained model (theta matrix).
    """

    n = int(np.log2(pD.size))

    if init is None:
        init = ModelConstruction.learn_indep(pD)

    pth_space = Likelihood.generate_pTh(init)

    opt = minimize(fun=score_func, x0=init, args=(pD, lam, n, pth_space), method="BFGS", jac=jacobi,
                   options={'maxiter': maxit, 'disp': trace, 'gtol': reltol}, callback=callback)

    theta = opt.x.reshape((n, n))

    if round_result:
        theta = np.around(theta, decimals=2)

    return theta
