# by Stefan Vocht
#
# this script implements the RegularizedOptimization.R in python

import numpy as np
from scipy.optimize import minimize

from . import Likelihood
from . import ModelConstruction

from typing import Callable


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


def score_reg(theta: np.ndarray, pD: np.ndarray, lam: float, n: int = None, pth_space: np.ndarray = None) -> float:
    """
    Score with L1 - regularization

    :param theta:
    :param pD: distribution given by the training data
    :param lam: tuning parameter for regularization
    :param n: number of columns/rows of theta
    :param pth_space: opional, with this parameter we can communicate with the gradient function and use pth there again -> performance boost
    :return:
    """
    n = n or int(np.sqrt(theta.size))
    theta = theta.reshape((n, n))

    return -(Likelihood.score(theta, pD, pth_space) - lam * L1(theta))


def grad_reg(theta: np.ndarray, pD: np.ndarray, lam: float, n: int = 0, pth_space: np.ndarray = None) -> np.ndarray:
    """
    Gradient with L1 - regularization

    :param theta:
    :param pD: distribution given by the training data
    :param lam: tuning parameter for regularization
    :param n: number of columns/rows of theta
    :param pth_space: opional, as pth is calculated in the score function anyways, we do not need to calculate it again -> performance boost
    :return:
    """
    n = n or int(np.sqrt(theta.size))
    theta_ = theta.reshape((n, n))

    return -(Likelihood.grad(theta_, pD, pth_space) - lam * L1_(theta_)).flatten()


def learn_MHN(pD: np.ndarray, init: np.ndarray = None, lam: float = 0, maxit: int = 5000,
              trace: bool = False, reltol: float = 1e-07, round_result: bool = True,
              callback: Callable = None, score_func: Callable = score_reg, jacobi: Callable = grad_reg) -> np.ndarray:
    """
    This function is used to train a MHN to a given probability distribution pD

    :param pD: probability distribution used to train the new model
    :param init: starting point for the training (initial theta)
    :param lam: tuning parameter for regularization
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

    theta = opt.x.reshape((n, n))

    if round_result:
        theta = np.around(theta, decimals=2)

    return theta


if __name__ == '__main__':
    m = np.arange(1, 10).reshape(3, 3).astype(np.float64)
    xx = np.arange(1, 9).astype(np.float64)

    print(L1(m))

    import timeit

    print(timeit.timeit(lambda: L1(m), number=10000))