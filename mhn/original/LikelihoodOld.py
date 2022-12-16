# by Stefan Vocht
#
# this script implements the Likelihood.R in python

import numpy as np
from numba import njit, prange

from . import ModelConstruction
from .PerformanceCriticalCodeOld import kron_vec, loop_j


@njit(cache=True)
def q_vec(theta: np.ndarray, x: np.ndarray, diag: bool = False, transp: bool = False) -> np.ndarray:
    """
    Multiplies the vector x with the matrix Q

    :param theta: thetas used to construct Q
    :param x: vector that is multiplied with Q
    :param diag: if False, the diagonal of Q is set to zero
    :param transp: if True, x is multiplied with Q^T

    :return: product of Q and x
    """
    n = theta.shape[1]
    y = np.zeros(2**n)

    for i in prange(n):
        y += kron_vec(theta, i, x, diag, transp)

    return y


@njit(cache=True)
def jacobi(theta: np.ndarray, b: np.ndarray, transp: bool = False) -> np.ndarray:
    """
    Returns the solution for [I - Q]^-1 x = b

    :param theta: thetas used to construct Q
    :param b:
    :param transp: if True, returns solution for ([I - Q]^-1)^T x = b
    :return:
    """
    n = theta.shape[1]

    x = np.full(2**n, 1 / 2**n)

    dg = 1 - ModelConstruction.q_diag(theta)

    for _ in range(n+1):
        x = b + q_vec(theta, x, diag=False, transp=transp)
        x = x / dg

    return x


@njit(cache=True)
def jacobi_eq11(theta: np.ndarray, b: np.ndarray, transp: bool = False) -> np.ndarray:
    """
    Implementation of equation 11 in the MHN paper (not perfectly working yet)

    :param theta: thetas used to construct Q
    :param b:
    :param transp: if True, returns solution for ([I - Q]^-1)^T x = b
    :return: Returns the solution for [I - Q]^-1 x = b
    """
    n = theta.shape[0]

    dg_inv = 1 / (1 - ModelConstruction.q_diag(theta))

    x = dg_inv * b
    sum = x.copy()

    for k in range(n):
        x = -dg_inv * q_vec(theta, x, diag=False, transp=transp)
        sum += x

    # @TODO using the abs function here should not be necessary, should be fixed
    return np.abs(sum)


@njit(cache=True)
def generate_pTh(theta: np.ndarray, p0: np.ndarray = None) -> np.ndarray:
    """
    Returns the probability distribution given by theta

    :param theta:
    :param p0:
    :return:
    """
    n = theta.shape[1]

    if p0 is None:
        p0 = np.zeros(2**n)
        p0[0] = 1

    return jacobi(theta, p0)


def score(theta: np.ndarray, pD: np.ndarray, pth_space: np.ndarray = None) -> float:
    """
    Calculates the score for the current theta

    :param theta:
    :param pD: probability distribution in the data
    :param pth_space: opional, with this parameter we can communicate with the function grad and use pth there again -> performance boost
    :return: score value
    """
    pth = generate_pTh(theta)

    if pth_space is not None:
        pth_space[:] = pth

    return pD.dot(np.log(pth))


@njit(cache=True)
def grad(theta: np.ndarray, pD: np.ndarray, pth_space: np.ndarray = None) -> np.ndarray:
    """
    Implements gradient calculation of equation 7

    :param theta:
    :param pD: probability distribution of the training data
    :param pth: as pth is calculated in the score function anyways, we do not need to calculate it again
    :return: gradient you get from equation 7
    """
    n = int(np.sqrt(theta.size))
    theta = theta.reshape((n, n))

    # distribution you get from our current model Theta (pth ~ "p_theta")
    if pth_space is None:
        # start distribution p_0 where no gene is mutated yet
        p0 = np.zeros(2 ** n)
        p0[0] = 1
        pth = jacobi(theta, p0)

    else:
        pth = pth_space

    # should be (pD / pth)^T * R_theta^-1 from equation 7
    q = jacobi(theta, pD / pth, transp=True)

    g = np.zeros((n, n))

    for i in range(n):
        r_vec = q * kron_vec(theta, i, pth, diag=True)
        loop_j(i, n, r_vec, g)

    return g


if __name__ == '__main__':

    n_ = 20
    np.random.seed(1)
    theta_ = ModelConstruction.random_theta(n_, 0.5)
    x_ = np.arange(2**n_, dtype=np.float64)
    pth_ = generate_pTh(theta_)

    print(pth_)
    import time
    for _ in range(2):
        s = time.time()
        print(grad(theta_, np.ones(2**n_), pth_))
        print(time.time()-s)

