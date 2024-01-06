"""
This submodule implements Likelihood.R from the original implementation in Cython.

It contains functions to compute the log-likelihood score and its gradient without state-space restriction as
well as functions for matrix-vector multiplications with the transition rate matrix and [I-Q]^(-1).
There are also functions to compute the probability distribution represented by an MHN and to sample artificial data
from a given MHN.
"""
# author(s): Stefan Vocht

cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport exp

from .ModelConstruction cimport q_diag
from .PerformanceCriticalCode cimport internal_kron_vec, loop_j, compute_inverse

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
    Returns the solution for [I - Q] x = b

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
    cdef np.ndarray[np.double_t] dg = 1 - q_diag(theta)
    cdef np.ndarray[np.double_t] pth = np.empty(2**n, dtype=np.double)

    if p0 is None:
        p0 = np.zeros(2**n)
        p0[0] = 1

    compute_inverse(theta, dg, p0, pth, False)
    return pth


def score(double[:, :] theta, np.ndarray[np.double_t] pD, np.ndarray[np.double_t] pth_space = None) -> float:
    """
    Calculates the score for the current theta

    :param theta:
    :param pD: probability distribution in the data
    :param pth_space: optional, with this parameter we can communicate with the function grad and use pth there again -> performance boost
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
    :param pth: as pth is calculated in the score function anyway, we do not need to calculate it again
    :param pth_space: optional, with this parameter we can communicate with the function score and use pth here again -> performance boost
    :return: gradient you get from equation 7
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


@cython.wraparound(False)
@cython.boundscheck(False)
def sample_artificial_data(np.ndarray[np.double_t, ndim=2] theta, int sample_num) -> np.ndarray:
    """
    Returns artificial data sampled from an MHN. Random values are generated with numpy, use np.random.seed()
    to make results reproducible.

    :param theta: theta matrix representing an MHN
    :param sample_num: number of samples in the generated data
    :returns: 2d numpy array in which every row corresponds to a sample and each column to a gene
    """
    cdef int n = theta.shape[0]
    cdef np.ndarray[np.double_t, ndim=2] exp_theta = np.exp(theta)
    cdef np.ndarray[np.int32_t, ndim=2] art_data = np.zeros((sample_num, n), dtype=np.int32)

    cdef np.ndarray[np.int32_t] in_current_sample = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t] possible_gene_mutations = np.empty(n, dtype=np.int32)
    cdef np.ndarray[np.double_t] rates_from_current_state = np.empty(n, dtype=np.double)

    cdef int j, gene, mutated_gene
    cdef int sample_index, current_sample_length
    cdef double observation_time, current_time, passed_time
    cdef double sum_rates, rate, random_crit, accumulated_rate

    for sample_index in range(sample_num):
        in_current_sample[:] = 0
        current_sample_length = 0
        observation_time = np.random.exponential(1)
        current_time = 0.
        while current_sample_length < n:
            sum_rates = 0.
            j = 0
            for gene in range(n):
                if not in_current_sample[gene]:
                    possible_gene_mutations[j] = gene
                    rate = exp_theta[gene, gene]
                    for mutated_gene in range(n):
                        if in_current_sample[mutated_gene]:
                            rate *= exp_theta[gene, mutated_gene]
                    rates_from_current_state[j] = rate
                    sum_rates += rate
                    j += 1

            passed_time = np.random.exponential(1/sum_rates)
            current_time += passed_time
            if current_time > observation_time:
                break
            random_crit = np.random.random(1)[0] * sum_rates
            accumulated_rate = 0.
            for j in range(n - current_sample_length):
                rate = rates_from_current_state[j]
                gene = possible_gene_mutations[j]
                accumulated_rate += rate
                if random_crit <= accumulated_rate:
                    in_current_sample[gene] = 1
                    current_sample_length += 1
                    art_data[sample_index, gene] = 1
                    break
    return art_data


def compute_next_event_probs(np.ndarray[np.double_t, ndim=2] theta, np.ndarray[np.int_t, ndim=1] current_state, double observation_rate = 0) -> np.ndarray:
    """
    Compute the probability for each event that it will be the next one to occur given the current state.

    :param theta: theta matrix representing an MHN
    :param current_state: array representing the current state, each entry corresponds to an event being present (1) or not (0)
    :param observation_rate: rate of the observation event, by default set to 0 which means no observation before some other event occurs
    :returns: array that contains the probability for each event that it will be the next one to occur

    :raise ValueError: if the size of theta does not match the size of current_state, a ValueError is raised
    """
    cdef int n = theta.shape[1]  # use shape[1] to be compatible with OmegaMHN
    if n != current_state.shape[0]:
        raise ValueError(f"Number of events represented by theta ({n}) does not match the size of current_state ({current_state.shape[0]})")

    cdef np.ndarray[np.double_t, ndim=2] theta_copy = theta.copy()
    cdef np.ndarray[np.double_t, ndim=1] result = np.zeros(n, dtype=np.double)
    cdef int i, j
    cdef double rate

    for i in range(n):
        if current_state[i]:
            continue
        rate = theta[i, i]
        for j in range(n):
            if current_state[j]:
                rate += theta[i, j]
        result[i] = exp(rate)

    result /= result.sum() + observation_rate
    return result


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

    **It can only be used if the mhn package was compiled with CUDA.**

    :param theta: theta matrix representing the MHN
    :param pD: probability distribution according to the training data
    :returns: tuple containing the gradient and the score

    :raise RuntimeError: this function raises a RuntimeError if the mhn package was not compiled with CUDA
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

    **It can only be used if the mhn package was compiled with CUDA.**

    :param theta: theta matrix representing the MHN
    :param b: vector that is multiplied with [I-Q]^(-1)
    :param transp: if set to True, computes solution for [I-Q]^T x = b
    :returns: the solution for [I-Q] x = b

    :raise RuntimeError: this function raises a RuntimeError if the mhn package was not compiled with CUDA
    """

    IF NVCC_AVAILABLE:
        cdef int n = theta.shape[0]
        cdef int nx = 1 << n
        cdef np.ndarray[np.double_t] xout = np.empty(nx, dtype=np.double)
        gpu_compute_inverse(&theta[0, 0], n, &b[0], &xout[0], transp)
        return xout
    ELSE:
        raise RuntimeError('The mhn package was not compiled with CUDA, so you cannot use this function.')