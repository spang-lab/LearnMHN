# by Stefan Vocht
#
# this script contains functions to compute the fisher matrix and gradients
#

import numpy as np
from numpy.linalg import det, inv, norm

from numba import njit

from mhn.original import Likelihood, PerformanceCriticalCode
from mhn.original import ModelConstruction
from mhn.original import UtilityFunctions


@njit
def compute_fisher_efficient(theta: np.ndarray, pth: np.ndarray = None) -> np.ndarray:
    """
    Computes the fisher matrix with a more efficient algorithm

    :param theta: matrix containing the theta values
    :param pth: optional parameter, containing the probability distribution yielded by theta

    :returns: the Fisher information matrix for theta
    """
    if pth is None:
        pth_ = Likelihood.generate_pTh(theta)
    else:
        pth_ = pth

    n = theta.shape[0]
    fisher_mat = np.zeros((n**2, n**2))
    ind_x = -1

    for s in range(n):
        # kron_vec function from the original MHN implementation
        dQ_st = PerformanceCriticalCode.kron_vec(theta, s, pth_, diag=True)
        zero_mask = np.arange(2**n)
        for t in range(n):
            # the derivative of Q wrt. theta_st (here stored in masked_dQ_st) mainly depends on s
            # the difference between the theta_st is only that for s != t some entries are set to zero
            # this is done here using zero_mask
            if s != t:
                masked_dQ_st = dQ_st * (zero_mask % 2)
            else:
                masked_dQ_st = dQ_st

            zero_mask //= 2

            # q_st described in section 4.1.1
            # the jacobi function computes the matrix-vector product [I-Q]^(-1) * b
            q_st = Likelihood.jacobi(theta, masked_dQ_st)
            q_st /= pth_
            q_st = Likelihood.jacobi(theta, q_st, transp=True)

            ind_x += 1
            ind_y = s * n
            # fisher matrix is symmetric, compute only the upper half
            # and use the shuffle trick for more efficiency
            for i in range(s, n):
                dQ_ij = PerformanceCriticalCode.kron_vec(theta, i, pth_, diag=True)
                dQ_ij *= q_st
                for j in range(n):
                    # reshaping for shuffling
                    dQ_ij_mat = dQ_ij.reshape((2**(n-1), 2))

                    # compute only the upper half
                    if ind_y >= ind_x:
                        if j == i:
                            fisher_mat[ind_x, ind_y] = dQ_ij.sum()
                        else:
                            fisher_mat[ind_x, ind_y] = dQ_ij_mat[:, 1].sum()

                    ind_y += 1
                    dQ_ij = dQ_ij_mat.T.flatten()

    # fill the lower triangular part
    fisher_mat += np.tril(fisher_mat.T, -1)

    return fisher_mat


@njit
def derivative_pth(theta: np.ndarray, pth: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Computes the derivative of pth wrt. theta_ij
    """

    right_part = PerformanceCriticalCode.kron_vec(theta, i, pth, diag=True)

    if i != j:
        n = theta.shape[0]
        indx = np.arange(2 ** n)
        indx //= 2 ** j

        right_part *= indx % 2

    return Likelihood.jacobi(theta, right_part)


def numerical_deriv_pth(theta: np.ndarray, pth: np.ndarray, i: int, j: int, h: float = 1e-10) -> np.ndarray:
    """
    Computes the derivative of pth wrt. theta_ij numerically

    :param theta: theta matrix
    :param pth: pth we get from theta
    :param h: small deviation used to compute the derivative
    :return: derivative
    """
    theta_new = theta.copy()
    theta_new[i, j] += h

    pth_new = Likelihood.generate_pTh(theta_new)

    return (pth_new - pth) / h


@njit
def compute_fisher(theta: np.ndarray, pth: np.ndarray = None) -> np.ndarray:
    """
    Computes the fisher matrix given a theta (not very efficient)

    :param theta:
    :param pth: Optional, may save some time if pth is already computed elsewhere to use it here again
    :return:
    """

    n = theta.shape[0]

    if pth is None:
        pth = Likelihood.generate_pTh(theta)

    fisher_matrix = np.empty((n ** 2, n ** 2))

    for i in range(n ** 2):
        p1 = derivative_pth(theta, pth, i // n, i % n)
        for j in range(n ** 2):
            p2 = derivative_pth(theta, pth, j // n, j % n)

            fisher_matrix[i, j] = np.sum(p1 * p2 / pth)

    return fisher_matrix


def fisher_test(theta: np.ndarray, pth: np.ndarray, fisher_theta: np.ndarray):
    """
    This is a small test function that tests if the Fisher matrix was computed correctly by testing if it is invariant
    to reparametrization

    :param theta: theta
    :param pth: probability distribution yielded by theta
    :param fisher_theta: the Fisher matrix for theta
    :return:
    """
    n = int(np.log2(pth.size))

    np.random.seed(1)

    theta_new = theta + np.random.rand(*theta.shape) / 100000000
    pth_new = Likelihood.generate_pTh(theta_new)

    d = theta_new.flatten() - theta.flatten()
    g = pth_new - pth

    return d.dot(fisher_theta.dot(d)), g.dot(g / pth), \
           d.dot(fisher_theta.dot(d)) / g.dot(g / pth), g.dot(g / pth) / d.dot(fisher_theta.dot(d)), \
           d.dot(d), g.dot(g), d.dot(d) / g.dot(g)


def mirror_logistic_func(x: float) -> float:
    return 1 / (np.exp(x) + 1)


def L2(theta: np.ndarray, eps: float = 1e-5) -> float:
    """
    Implements the L2 regularization for the MHN used in natural gradient descent

    :param theta:
    :param eps: prevents division by zero
    :return:
    """
    return np.sum(theta**2)


def L2_(theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Derivative of L2 regularization

    :param theta:
    :param eps:
    :return:
    """
    return 2 * theta


def learn_MHN_with_NGD(pD: np.ndarray, init: np.ndarray = None, maxit: int = 5000, reltol: float = 1e-07,
                       round_result: bool = True, eta: float = 0.02, callback=None, verbose=False) -> np.ndarray:
    """
    This function learns a MHN using the Natrual Gradient Descent.

    :param pD: probability distribution of the training data
    :param init: start value for training, default starting point is the independence model
    :param maxit: maximum number of iterations
    :param reltol: stop learning when the score (the KL-divergence) is smaller than this value
    :param round_result: if True, round the resulting Theta values
    :param eta: scaling factor for the gradient descent
    :param callback: Optional, a function that takes a numpy array (theta) as parameter and is called in each iteration
    :param verbose: if True, prints current KL-divergence and step size in each iteration

    :return: Theta representing the trained MHN
    """

    # get size of the Theta matrix
    n = int(np.log2(pD.size))

    if init is None:
        theta = ModelConstruction.learn_indep(pD)
    else:
        theta = init.reshape((n, n))

    pth = Likelihood.generate_pTh(theta)

    # if the KL-divergence between the current distribution and the data distribution is smaller than reltol,
    # return the initial model
    if UtilityFunctions.KL_div(pD, pth) < reltol:
        return theta

    for k in range(1, maxit):
        # call the callback function in each iteration if given
        if callback is not None:
            callback(theta.flatten())

        # pth might contain NaNs if the values in Theta get too large
        if np.any(np.isnan(pth)):
            raise ValueError("pth contains NaN!")

        # compute the normal gradient
        gradient = - Likelihood.grad(theta, pD, pth)
        gradient = gradient.flatten()

        # compute the Fisher information matrix
        fisher = compute_fisher_efficient(theta, pth)

        # we need to compute the inverse of the Fisher information matrix
        # and we have to make sure that the matrix is invertible
        try:
            fisher_inv = inv(fisher)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError('Fisher information matrix is singular')

        # compute the next step by first multiplying the inverse of the Fisher with the gradient
        step = fisher_inv.dot(gradient)
        # scale the step according to the formula we derived in the NGD section and reshape the result
        step = np.sqrt(eta / (gradient.dot(fisher_inv).dot(gradient))) * step.reshape((n, n))

        if verbose:
            print(f"{k}, KL-divergence: {UtilityFunctions.KL_div(pD, pth):.7f}, step size: {norm(step):.3f}")

        # update the current theta
        theta -= step
        pth = Likelihood.generate_pTh(theta)

        # if the KL-divergence between the current distribution and the data distribution is smaller than reltol, break
        if UtilityFunctions.KL_div(pD, pth) < reltol:
            break

    if round_result:
        theta = np.around(theta, decimals=2)

    return theta
