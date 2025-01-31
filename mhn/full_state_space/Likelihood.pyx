"""
This submodule implements Likelihood.R from the original implementation in Cython.

It contains functions to compute the log-likelihood score and its gradient without state-space restriction as
well as functions for matrix-vector multiplications with the transition rate matrix and [I-Q]^(-1).
There are also functions to compute the probability distribution represented by an cMHN.
"""
# author(s): Stefan Vocht

cimport cython

from libc.stdlib cimport malloc, free

from .ModelConstruction cimport q_diag
from .PerformanceCriticalCode cimport internal_kron_vec, loop_j, compute_inverse

from scipy.linalg.cython_blas cimport dcopy, dscal, daxpy, ddot

import numpy as np
cimport numpy as np

np.import_array()

cdef internal_q_vec(double[:, :] theta, double[:] x, double[:] yout, bint diag = False, bint transp = False):
    """
    Multiplies the vector x with the transition rate matrix Q.

    Args:
        theta (double[:, :]): Thetas used to construct the transition rate matrix Q.
        x (double[:]): Vector that is multiplied with Q.
        yout (double[:]): Vector that will contain the result of the multiplication.
        diag (bool, optional): If False, the diagonal of Q is set to zero. Defaults to False.
        transp (bool, optional): If True, x is multiplied with Q^T (transposed Q). Defaults to False.

    Returns:
        double[:]: The product of Q and x, stored in yout.
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
    Multiplies the vector x with the transition rate matrix Q.

    Args:
        theta (np.ndarray[np.double_t]): Thetas used to construct the transition rate matrix Q.
        x (np.ndarray[np.double_t]): Vector that is multiplied with Q.
        diag (bool, optional): If False, the diagonal of Q is set to zero. Defaults to False.
        transp (bool, optional): If True, x is multiplied with Q^T (transposed Q). Defaults to False.

    Returns:
        np.ndarray[np.double_t]: The product of Q and x.
    """
    cdef int n = theta.shape[0]
    cdef np.ndarray[np.double_t] result = np.empty(2**n, dtype=np.double)
    internal_q_vec(theta, x, result, diag, transp)
    return result


cpdef np.ndarray[np.double_t] jacobi(double[:, :] theta, np.ndarray[np.double_t] b, bint transp = False):
    """
    Returns the solution for [I - Q] x = b.

    Args:
        theta (np.ndarray[np.double_t]): Thetas used to construct the transition rate matrix Q.
        b (np.ndarray[np.double_t]): The vector on the right-hand side of the equation.
        transp (bool, optional): If True, returns the solution for ([I - Q]^-1)^T x = b. Defaults to False.

    Returns:
        np.ndarray[np.double_t]: The solution vector x for the equation [I - Q] x = b.
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
    Returns the probability distribution given by theta.

    Args:
        theta (np.ndarray[np.double_t]): The matrix representing the cMHN parameters.
        p0 (optional): The initial probability distribution. If None, it assumes no initial active events. Defaults to None.

    Returns:
        np.ndarray[np.double_t]: The probability distribution computed from the given theta.
    """
    cdef int n = theta.shape[1]
    cdef np.ndarray[np.double_t] dg = 1 - q_diag(theta)
    cdef np.ndarray[np.double_t] pth = np.empty(2**n, dtype=np.double)

    if p0 is None:
        p0 = np.zeros(2**n)
        p0[0] = 1

    compute_inverse(theta, dg, p0, pth, False)
    return pth


def score(double[:, :] theta, np.ndarray[np.double_t] pD, np.ndarray[np.double_t] pth_space = None) -> float:
    """
    Calculates the score for the current theta.

    Args:
        theta (np.ndarray[np.double_t]): The matrix representing the cMHN parameters.
        pD (np.ndarray[np.double_t]): The probability distribution in the data.
        pth_space (optional, np.ndarray[np.double_t]): Optional parameter for performance optimization. If provided, it allows communication with
                                                       the gradient function, reducing computation time. Defaults to None.

    Returns:
        float: The score value for the given theta.
    """
    cdef np.ndarray[np.double_t] pth = generate_pTh(theta)

    if pth_space is not None:
        pth_space[:] = pth

    return pD.dot(np.log(pth))


def grad(double[:, :] theta, np.ndarray[np.double_t] pD, np.ndarray[np.double_t] pth_space = None) -> np.ndarray:
    """
    Implements the gradient calculation of equation 7.

    Args:
        theta (np.ndarray[np.double_t]): The matrix representing the cMHN parameters.
        pD (np.ndarray[np.double_t]): The probability distribution in the data.
        pth_space (optional, np.ndarray[np.double_t]): Optional parameter for performance optimization. If provided, it allows communication with
                                                       the score function, reducing computation time. Defaults to None.

    Returns:
        np.ndarray[np.double_t]: The gradient calculated according to equation 7.
    """
    cdef int n = theta.shape[0]
    cdef int nx = 1 << n
    cdef int i, j

    cdef np.ndarray[np.double_t] p0, pth

    cdef np.ndarray[np.double_t] dg = 1 - q_diag(theta)

    # distribution you get from our current model Theta (pth ~ "p_theta")
    if pth_space is None:
        # start distribution p_0 where no gene is mutated yet
        p0 = np.zeros(2 ** n)
        p0[0] = 1
        pth = np.empty(2**n, dtype=np.double)
        compute_inverse(theta, dg, p0, pth, False)

    else:
        pth = pth_space

    # should be (pD / pth)^T * R_theta^-1 from equation 7
    cdef np.ndarray[np.double_t] q = np.empty(2**n, dtype=np.double)
    compute_inverse(theta, dg, pD / pth, q, True)

    cdef np.ndarray[np.double_t, ndim=2] g = np.zeros((n, n))
    cdef double *r_vec = <double *> malloc(nx * sizeof(double))

    for i in range(n):
        internal_kron_vec(theta, i, pth, r_vec, diag=True, transp=False)
        for j in range(nx):
            r_vec[j] *= q[j]
        loop_j(i, n, r_vec, &g[0, 0])

    free(<void *> r_vec)
    return g


IF NVCC_AVAILABLE:
    cdef extern from *:
        """
        #ifdef _WIN32
        #define DLL_PREFIX __declspec(dllexport)
        #else
        #define DLL_PREFIX
        #endif

        int DLL_PREFIX cuda_full_state_space_gradient_score(double *ptheta, int n, double *pD, double *grad_out, double *score_out);
        void DLL_PREFIX gpu_compute_inverse(double *theta, int n, double *b, double *xout, int transp);
        """
        int cuda_full_state_space_gradient_score(double *ptheta, int n, double *pD, double *grad_out, double *score_out)
        void gpu_compute_inverse(double *theta, int n, double *b, double *xout, int transp)


def cuda_gradient_and_score(double[:, :] theta, double[:] pD):
    """
    Computes the log-likelihood score and its gradient for a given theta and a given empirical distribution.

    **This function can only be used if the mhn package was compiled with CUDA.**

    Args:
        theta (np.ndarray[np.double_t]): The theta matrix representing the cMHN.
        pD (np.ndarray[np.double_t]): The probability distribution according to the training data.

    Returns:
        tuple: The gradient and the log-likelihood score.

    Raises:
        RuntimeError: If the mhn package was not compiled with CUDA.
    """

    IF NVCC_AVAILABLE:
        cdef int n = theta.shape[0]
        cdef double score
        cdef np.ndarray[np.double_t, ndim=2] gradient = np.empty((n, n), dtype=np.double)
        cdef int error_code

        error_code = cuda_full_state_space_gradient_score(&theta[0, 0], n, &pD[0], &gradient[0, 0], &score)

        return gradient, score
    ELSE:
        raise RuntimeError('The mhn package was not compiled with CUDA, so you cannot use this function.')


def cuda_compute_inverse(double[:, :] theta, double[:] b, bint transp = False):
    """
    Computes the solution for [I-Q] x = b on the GPU.

    **This function can only be used if the mhn package was compiled with CUDA.**

    Args:
        theta (np.ndarray[np.double_t]): The theta matrix representing the cMHN.
        b (np.ndarray[np.double_t]): A vector that is multiplied with [I-Q]^(-1).
        transp (bool, optional): If set to True, computes the solution for [I-Q]^T x = b. Defaults to False.

    Returns:
        np.ndarray[np.double_t]: The solution for [I-Q] x = b.

    Raises:
        RuntimeError: If the mhn package was not compiled with CUDA.
    """

    IF NVCC_AVAILABLE:
        cdef int n = theta.shape[0]
        cdef int nx = 1 << n
        cdef np.ndarray[np.double_t] xout = np.empty(nx, dtype=np.double)
        gpu_compute_inverse(&theta[0, 0], n, &b[0], &xout[0], transp)
        return xout
    ELSE:
        raise RuntimeError('The mhn package was not compiled with CUDA, so you cannot use this function.')