# by Stefan Vocht
#
# this script implements Likelihood.R in Cython
#

cimport cython

from libc.stdlib cimport malloc, free

from .ModelConstruction cimport q_diag
from .PerformanceCriticalCode cimport internal_kron_vec, loop_j

from scipy.linalg.cython_blas cimport dcopy, dscal, daxpy, ddot

import numpy as np
cimport numpy as np

np.import_array()

cdef internal_q_vec(double[:, :] theta, double[:] x, double[:] yout, bint diag = False, bint transp = False):
    """
    Multiplies the vector x with the matrix Q

    :param theta: thetas used to construct Q
    :param x: vector that is multiplied with Q
    :param yout: vector that contains the result of the multiplication
    :param diag: if False, the diagonal of Q is set to zero
    :param transp: if True, x is multiplied with Q^T

    :return: product of Q and x
    """
    cdef int n = theta.shape[0]
    cdef int nx = x.shape[0]
    cdef int i
    cdef double one_d = 1
    cdef int one_i = 1
    cdef double zero = 0

    cdef double *result_vec = <double *> malloc(sizeof(double) * nx)

    # initialize yout with zero
    dscal(&nx, &zero, &yout[0], &one_i)

    for i in range(n):
        internal_kron_vec(theta, i, x, result_vec, diag, transp)
        # add result of restricted_kronvec to yout
        daxpy(&nx, &one_d, result_vec, &one_i, &yout[0], &one_i)

    free(result_vec)


cpdef np.ndarray[np.double_t] q_vec(double[:, :] theta, double[:] x, bint diag = False, bint transp = False):
    """
    Multiplies the vector x with the matrix Q

    :param theta: thetas used to construct Q
    :param x: vector that is multiplied with Q
    :param diag: if False, the diagonal of Q is set to zero
    :param transp: if True, x is multiplied with Q^T

    :return: product of Q and x
    """
    cdef int n = theta.shape[0]
    cdef np.ndarray[np.double_t] result = np.empty(2**n, dtype=np.double)
    internal_q_vec(theta, x, result, diag, transp)
    return result


cpdef np.ndarray[np.double_t] jacobi(double[:, :] theta, np.ndarray[np.double_t] b, bint transp = False):
    """
    Returns the solution for [I - Q]^-1 x = b

    :param theta: thetas used to construct Q
    :param b:
    :param transp: if True, returns solution for ([I - Q]^-1)^T x = b
    :return:
    """
    cdef int n = theta.shape[1]
    cdef int i

    cdef np.ndarray[np.double_t] x = np.full(2**n, 1. / (2**n), dtype=np.double)
    cdef np.ndarray[np.double_t] dg = 1 - q_diag(theta)

    for i in range(n+1):
        x = b + q_vec(theta, x, diag=False, transp=transp)
        x = x / dg

    return x


cpdef np.ndarray[np.double_t] generate_pTh(double[:, :] theta, p0 = None):
    """
    Returns the probability distribution given by theta

    :param theta:
    :param p0:
    :return:
    """
    cdef int n = theta.shape[1]

    if p0 is None:
        p0 = np.zeros(2**n)
        p0[0] = 1

    return jacobi(theta, p0)


def score(double[:, :] theta, np.ndarray[np.double_t] pD, np.ndarray[np.double_t] pth_space = None) -> float:
    """
    Calculates the score for the current theta

    :param theta:
    :param pD: probability distribution in the data
    :param pth_space: opional, with this parameter we can communicate with the function grad and use pth there again -> performance boost
    :return: score value
    """
    cdef np.ndarray[np.double_t] pth = generate_pTh(theta)

    if pth_space is not None:
        pth_space[:] = pth

    return pD.dot(np.log(pth))


def grad(double[:, :] theta, np.ndarray[np.double_t] pD, np.ndarray[np.double_t] pth_space = None) -> np.ndarray:
    """
    Implements gradient calculation of equation 7

    :param theta:
    :param pD: probability distribution of the training data
    :param pth: as pth is calculated in the score function anyways, we do not need to calculate it again
    :param pth_space: opional, with this parameter we can communicate with the function score and use pth here again -> performance boost
    :return: gradient you get from equation 7
    """
    cdef int n = theta.shape[0]
    cdef int nx = 1 << n
    cdef int i, j

    cdef np.ndarray[np.double_t] p0, pth

    # distribution you get from our current model Theta (pth ~ "p_theta")
    if pth_space is None:
        # start distribution p_0 where no gene is mutated yet
        p0 = np.zeros(2 ** n)
        p0[0] = 1
        pth = jacobi(theta, p0)

    else:
        pth = pth_space

    # should be (pD / pth)^T * R_theta^-1 from equation 7
    cdef np.ndarray[np.double_t] q = jacobi(theta, pD / pth, transp=True)

    cdef np.ndarray[np.double_t, ndim=2] g = np.zeros((n, n))
    cdef double *r_vec = <double *> malloc(nx * sizeof(double))

    for i in range(n):
        internal_kron_vec(theta, i, pth, r_vec, diag=True, transp=False)
        for j in range(nx):
            r_vec[j] *= q[j]
        loop_j(i, n, r_vec, &g[0, 0])

    free(<void *> r_vec)
    return g