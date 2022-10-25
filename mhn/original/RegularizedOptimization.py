"""
This submodule implements RegularizedOptimization.R in Python.

It contains functions to learn an MHN on the *full state-space* for a given data distribution and implements the L1 regularization.
"""
# author(s): Stefan Vocht

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from . import Likelihood
from . import ModelConstruction

from typing import Callable


def L1(theta: np.ndarray, eps: float = 1e-05) -> float:
    """
    Computes the L1 penalty

    :param theta: the theta matrix representing the MHN
    :param eps: small epsilon value, mainly there for the derivative

    :returns: the L1 penalty for the given theta matrix
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return np.sum(np.sqrt(theta_**2 + eps))


def L1_(theta: np.ndarray, eps: float = 1e-05) -> np.ndarray:
    """
    Derivative of the L1 penalty

    :param theta: the theta matrix representing the MHN
    :param eps: small epsilon value that makes sure that we don't divide by zero

    :returns: the derivative of the L1 penalty
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return theta_ / np.sqrt(theta_**2 + eps)


def score_reg(theta: np.ndarray, pD: np.ndarray, lam: float, n: int = None, pth_space: np.ndarray = None) -> float:
    """
    Score with L1 - regularization

    :param theta: the theta matrix representing the MHN
    :param pD: distribution given by the training data
    :param lam: tuning parameter lambda for regularization
    :param n: number of columns/rows of theta
    :param pth_space: optional, with this parameter we can communicate with the gradient function and use pth there again -> performance boost

    :returns: the score of the current MHN penalized with the L1 regularization
    """
    n = n or int(np.sqrt(theta.size))
    theta = theta.reshape((n, n))

    return -(Likelihood.score(theta, pD, pth_space) - lam * L1(theta))


def grad_reg(theta: np.ndarray, pD: np.ndarray, lam: float, n: int = 0, pth_space: np.ndarray = None) -> np.ndarray:
    """
    Gradient with L1 - regularization

    :param theta: the theta matrix representing the MHN
    :param pD: distribution given by the training data
    :param lam: tuning parameter lambda for regularization
    :param n: number of columns/rows of theta
    :param pth_space: optional, as pth is calculated in the score function anyway, we do not need to calculate it again -> performance boost

    :return: the gradient of the L1 - regularized score
    """
    n = n or int(np.sqrt(theta.size))
    theta_ = theta.reshape((n, n))

    return -(Likelihood.grad(theta_, pD, pth_space) - lam * L1_(theta_)).flatten()


def learn_MHN(pD: np.ndarray, init: np.ndarray = None, lam: float = 0, maxit: int = 5000,
              trace: bool = False, reltol: float = 1e-07, round_result: bool = True,
              callback: Callable = None, score_func: Callable = score_reg, jacobi: Callable = grad_reg) -> OptimizeResult:
    """
    This function is used to train an MHN to a given probability distribution pD.

    :param pD: probability distribution used to train the new model
    :param init: starting point for the training (initial theta)
    :param lam: tuning parameter lambda for regularization
    :param maxit: maximum number of training iterations
    :param trace: set to True to print convergence messages (see scipy.optimize.minimize)
    :param reltol: Gradient norm must be less than reltol before successful termination (see "gtol" scipy.optimize.minimize)
    :param round_result: if True, the result is rounded to two decimal places
    :param callback: function called after each iteration, must take theta as argument
    :param score_func: score function used for training
    :param jacobi: gradient function used for training
    :return: trained model
    """

    n = int(np.log2(pD.size))

    if init is None:
        init = ModelConstruction.learn_indep(pD)

    pth_space = Likelihood.generate_pTh(init)

    opt = minimize(fun=score_func, x0=init, args=(pD, lam, n, pth_space), method="BFGS", jac=jacobi,
                   options={'maxiter': maxit, 'disp': trace, 'gtol': reltol}, callback=callback)

    opt.x = opt.x.reshape((n, n))

    if round_result:
        opt.x = np.around(opt.x, decimals=2)

    return opt
