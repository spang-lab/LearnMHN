# by Stefan Vocht
#
# here we implement the gradient approximation by Gotovos et al. (2021)
# in Cython
#

#cython: language_level=3

cimport cython

from libc.stdlib cimport malloc, free, rand, srand, RAND_MAX
from libc.math cimport exp 

from mhn.ssr.state_storage cimport StateStorage

import numpy as np
cimport numpy as cnp

import random

cnp.import_array()


# as even the Cython implementation was too slow, the functions were also implemented in pure C
cdef extern from "c_approximate_gradient.c":
    double gradient_and_score_c(double *theta, int n, State *mutation_data, int data_size, int m, int burn_in_samples, double *grad_out)
    void set_c_seed(int seed)
    ctypedef struct State:
        unsigned int parts[STATE_SIZE]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double q_next(double[:, :] theta, int[:] curr_sequence, int new_element):
    """
    This function computes q_{sigma_[i-1] -> sigma_[i]} as used in eq. 6

    :param theta: theta matrix (in exponential form!) parametrizing the MHN
    :param curr_sequence: sigma_[i-1], sequence before adding the new element
    :param new_element: new element to be added to the given sequence
    :return:
    """
    cdef double result = 1.
    cdef int i

    for i in range(curr_sequence.shape[0]):
        result *= theta[new_element, curr_sequence[i]]

    result *= theta[new_element, new_element]
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double[:] q_next_deriv(double[:, :] theta, int[:] curr_sequence, int new_element, int i):
    """
    Computes the derivative of q_next for all theta_i*
    """
    cdef double[:] result = np.zeros(theta.shape[0])
    cdef int j

    if i != new_element:
        return result

    cdef double q_n = q_next(theta, curr_sequence, new_element)

    result[i] = q_n

    for j in range(curr_sequence.shape[0]):
        result[curr_sequence[j]] = q_n

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double q_tilde(double[:, :] theta, int[:] sequence):
    """
    This function computes the q with a tilde used in eq. 6, which represents a diagonal element of Q

    :param theta: theta matrix (in exponential form!) parametrizing the MHN
    :param sequence: sequence of mutated genes (sorted!)
    :return:
    """
    cdef double result = 0
    cdef int i, j
    cdef int n = theta.shape[0]
    cdef double r_loc

    cdef short* in_sequence = <short *> malloc(n * sizeof(short))

    for i in range(n):
        in_sequence[i] = 0

    for i in range(sequence.shape[0]):
        in_sequence[sequence[i]] = 1

    for i in range(n):
        if not in_sequence[i]:
            r_loc = 1
            for j in range(n):
                if in_sequence[j]:
                    r_loc *= theta[i, j]
            r_loc *= theta[i, i]
            result += r_loc

    free(in_sequence)
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double[:] q_tilde_deriv(double[:, :] theta, int[:] sequence, int i):
    """
    Computes the derivate of q_tilde for all theta_i*
    """

    cdef double[:] result = np.zeros(theta.shape[0])
    cdef int j
    cdef int seq_length = sequence.shape[0]

    for j in range(seq_length):
        if sequence[j] == i:
            return result

    cdef double q_n = q_next(theta, sequence, i)

    result[i] = q_n

    for j in range(seq_length):
        result[sequence[j]] = q_n

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double p_sigma(double[:, :] theta, int[:] sequence):
    """
    Computes the probability to observe the a sequence sigma according to eq. 6

    :param theta: theta matrix parametrizing the MHN
    :param sequence: sequence of mutated genes
    :return:
    """

    cdef double result = 1
    cdef int i, j

    for i in range(sequence.shape[0]):
        result *= q_next(theta, sequence[:i], sequence[i])
        result /= 1 + q_tilde(theta, sequence[:i])

    result /= 1 + q_tilde(theta, sequence)
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:] p_sigma_deriv(double[:, :] theta, int[:] sequence, int k):
    """
    Computes the derivative of p_sigma for all theta_k*
    """

    cdef int n = theta.shape[0]
    cdef double[:] result = np.zeros(n)
    cdef double p_sig = p_sigma(theta, sequence)

    cdef double[:] q_next_der, q_tilde_der

    cdef int i, j
    cdef double p_loc, one_plus_tilde, q_n

    for i in range(sequence.shape[0]):
        p_loc = p_sig
        one_plus_tilde = 1 + q_tilde(theta, sequence[:i])
        q_n = q_next(theta, sequence[:i], sequence[i])
        # one_plus_tilde is squared in denominator of deriv, but partially compensated by removing the corresponding factor in p_sig
        p_loc /= one_plus_tilde
        p_loc /= q_n

        q_next_der = q_next_deriv(theta, sequence[:i], sequence[i], k)
        q_tilde_der = q_tilde_deriv(theta, sequence[:i], k)

        for j in range(n):
            result[j] += p_loc * (one_plus_tilde * q_next_der[j] - q_n * q_tilde_der[j])

    one_plus_tilde = 1 + q_tilde(theta, sequence)
    p_sig /= one_plus_tilde
    q_tilde_der = q_tilde_deriv(theta, sequence, k)

    for j in range(n):
        result[j] -= p_sig * q_tilde_der[j]

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double draw_from_q(double[:, :] theta, int[:] s, int[:] sigma_out):
    """
    Implementation of "Drawing from proposal Q" as shown in Appendix E of the paper

    :param theta: theta matrix (in exponential form!) parametrizing the MHN
    :param s: number array representing mutated genes
    :param sigma: array of same size as s, output for sampled path
    :return: new sequence sigma and Q_val
    """

    cdef int s_size = s.shape[0]

    cdef double q_val = 1  # in paper this is set to 0, but that makes no sense
    # cdef cnp.ndarray[cnp.int_t, ndim=1] sigma = np.empty(s_size, dtype=np.int)

    cdef int *s_without_sigma = <int *> malloc(s_size * sizeof(int))
    cdef double *u = <double *> malloc(s_size * sizeof(double))

    cdef int k, i, j, t, v, new_entry_index = 0
    cdef bint in_sigma
    cdef double dv, tmp_sum, sum_u, random_num

    for k in range(s_size):
        s_without_sigma[k] = s[k]

    for k in range(s_size):
        for i in range(s_size - k):
            v = s_without_sigma[i]
            sigma_out[k] = v
            dv = 1
            for j in range(theta.shape[0]):
                # check if j is in sigma
                in_sigma = False
                for t in range(k+1):
                    if sigma_out[t] == j:
                        in_sigma = True
                        break

                if in_sigma:
                    continue

                tmp_sum = theta[j, j]
                for t in range(k+1):
                    tmp_sum *= theta[j, sigma_out[t]]

                dv += tmp_sum

            tmp_sum = 1 / theta[v, v]
            for t in range(s_size - k):
                tmp_sum *= theta[s_without_sigma[t], v]

            u[i] = tmp_sum / dv

        sum_u = 0
        for t in range(s_size - k):
            sum_u += u[t]

        # random number between 0 and the sum of u
        random_num = rand() / (1.0 * RAND_MAX) * sum_u
        tmp_sum = 0
        for t in range(s_size - k):
            tmp_sum += u[t]
            if random_num <= tmp_sum:
                new_entry_index = t
                break

        sigma_out[k] = s_without_sigma[new_entry_index]
        q_val *= u[new_entry_index] / sum_u

        # remove new sigma entry from s_without_sigma
        for i in range(new_entry_index, s_size - k):
            s_without_sigma[i] = s_without_sigma[i+1]

    free(s_without_sigma)
    free(u)

    return q_val


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:, :] approx_gradient(double[:, :] theta, int state, int m = 50, int burn_in_samples = 10):
    """
    Implements the approximated gradient as shown in eq. 7

    :param theta:
    :param state: Integer where the 1s in binary form represent mutations
    :param m: number of samples taken for computing the gradient
    :param burn_in_samples: number of samples taken at the beginning without adding to the gradient
    :return: approximated gradient
    """

    cdef int n = theta.shape[0]
    cdef int i, j, k

    # matrix containing the final result
    cdef cnp.ndarray[cnp.float_t, ndim=2] resulting_gradient = np.zeros((n, n), dtype=np.float)
    cdef double[:] p_sigma_deriv_result

    cdef double q_val_old, q_val_new
    cdef int[:] sigma_old, sigma_new, tmp
    cdef double p_old, inv_p_old, p_new, p_accept

    # if there are no mutations, we do not need to sample paths but can directly return
    # the correct gradient for that state
    if state == 0:
        sigma_old = np.empty(0, dtype=np.int)
        inv_p_old = 1 / p_sigma(theta, sigma_old)
        for k in range(n):
            p_sigma_deriv_result = p_sigma_deriv(theta, sigma_old, k)
            for j in range(n):
                resulting_gradient[k, j] = inv_p_old * p_sigma_deriv_result[j]

        return resulting_gradient

    # allocate the memory for state_as_array, sigma_old and sigma_new
    cdef int *allocated_memory = <int *> malloc(3 * n * sizeof(int))

    # number of mutations in the given state
    cdef int mutation_num = 0

    # convert the state from bit form into an array containing all mutated genes
    # the result is stored at the later position of state_as_array
    for i in range(n):
        if (state >> i) & 1:
            allocated_memory[mutation_num] = i
            mutation_num += 1

    cdef int[:] state_as_array = <int[:mutation_num]> allocated_memory

    # sigmas represent paths from the origin to the given state
    sigma_old = <int[:mutation_num]> (allocated_memory + n)
    sigma_new = <int[:mutation_num]> (allocated_memory + n + n)

    # draw a random path at the beginning
    q_val_old = draw_from_q(theta, state_as_array, sigma_old)
    p_old = p_sigma(theta, sigma_old)

    # generate some burn in samples for better results
    for i in range(burn_in_samples):
        q_val_new = draw_from_q(theta, state_as_array, sigma_new)
        p_new = p_sigma(theta, sigma_new)
        p_accept = (p_new * q_val_old) / (p_old * q_val_new)

        if p_accept > 1:
            p_accept = 1.

        # accept the new path according to the acceptance probability
        if rand() <= p_accept * RAND_MAX:
            p_old = p_new
            tmp = sigma_old
            sigma_old = sigma_new
            sigma_new = tmp
            q_val_old = q_val_new

    # store the gradient from the current path in p_old_grad so we can use it again
    # if a new proposed path is rejected in the next round
    cdef double *p_old_grad = <double *> malloc(n * n * sizeof(double))

    inv_p_old = 1 / p_old

    # initilize p_old_grad with the gradient corresponding to the initial path
    for k in range(n):
        p_sigma_deriv_result = p_sigma_deriv(theta, sigma_old, k)
        for j in range(n):
            p_old_grad[k*n + j] = p_sigma_deriv_result[j]

    # sample paths and get their gradient
    for i in range(m):
        q_val_new = draw_from_q(theta, state_as_array, sigma_new)
        p_new = p_sigma(theta, sigma_new)
        p_accept = (p_new * q_val_old) / (p_old * q_val_new)

        if p_accept > 1:
            p_accept = 1.

        # accept the new path according to the acceptance probability
        if rand() <= p_accept * RAND_MAX:
            p_old = p_new
            inv_p_old = 1 / p_old
            tmp = sigma_old
            sigma_old = sigma_new
            sigma_new = tmp
            q_val_old = q_val_new

            for k in range(n):
                p_sigma_deriv_result = p_sigma_deriv(theta, sigma_old, k)
                for j in range(n):
                    p_old_grad[k*n + j] = p_sigma_deriv_result[j]

        for k in range(n):
            for j in range(n):
                resulting_gradient[k, j] += inv_p_old * p_old_grad[k*n + j]

    cdef double inv_m = 1 / (1. * m)
    for k in range(n):
        for j in range(n):
            resulting_gradient[k, j] *= inv_m

    free(p_old_grad)
    free(allocated_memory)

    return resulting_gradient


def gradient(double[:, :] theta, int[:] mutation_data, int m = 50, int burn_in_samples = 10) -> np.ndarray:
    """
    Computes the complete approximated gradient for given mutation data
    Note that this function takes the mutation_data as a numpy array, this is due to the fact that this function is not really
    used anymore as you should always use the pure c implementation as it is much faster.

    :param theta: current theta matrix
    :param mutation_data: list containing integers for each sample, where the integer represent the mutations in the sample
    :param m: number of samples taken for each approximate gradient
    :param burn_in_samples: burn-in-samples used in approximate gradient
    :return: gradient
    """

    cdef int n = theta.shape[0]
    cdef cnp.ndarray[cnp.float_t, ndim=2]  final_gradient = np.zeros((n, n), dtype=np.float)

    cdef int state
    for state in mutation_data:
        final_gradient += approx_gradient(theta, state, m, burn_in_samples)

    return final_gradient / len(mutation_data)


def gradient_and_score_using_c(double[:, :] theta, StateStorage mutation_data, int m = 50, int burn_in_samples = 10) -> np.ndarray:
    """
    This is a wrapper for the C implementation of the approximated gradient, so that it can be called from a Python script

    :param theta: matrix containing the theta entries
    :param mutation_data: StateStorage object containing the mutation data used to train the current MHN
    :param m: number of paths that should be sampled to approximate the gradients for each tumor sample
    :param burn_in_samples: number of paths that should be sampled at the beginning as burn in for better results
    """

    cdef int n = theta.shape[0]
    cdef int data_size = mutation_data.get_data_shape()[0]
    cdef cnp.ndarray[cnp.float_t, ndim=2]  final_gradient = np.zeros((n, n), dtype=np.float)
    cdef double score 

    score = gradient_and_score_c(&theta[0, 0], n, &mutation_data.states[0], data_size, m, burn_in_samples, &final_gradient[0, 0])

    return (final_gradient, score)


def set_seed(int seed):
    """
    Sets a seed to reproduce your results 
    """
    srand(seed)
    set_c_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def draw_one_sample(double[:, :] theta):
    """
    Draws a random sample according to the distribution yielded by the given MHN.
    This is "Algorithm 1" from the Gotovos et al. paper.

    :param theta: theta entries of the MHN
    """
    
    cdef double t_obs = np.random.exponential(1)
    cdef double t = 0
    cdef list s = list()
    cdef double h, sum_p, random_num
    cdef int s_length = 0
    cdef double q_tilde_val
    cdef int n = theta.shape[0]
    cdef int i, chosen_index
    cdef cnp.ndarray[cnp.int_t, ndim=1] s_as_array

    cdef double *p = <double *> malloc(n * sizeof(double))
    cdef list not_in_s = list(range(n))

    while t < t_obs and s_length < n:
        s_as_array = np.array(s, dtype=np.int)
        q_tilde_val = q_tilde(theta, s_as_array)
        # draw h
        h = np.random.exponential(1/q_tilde_val)

        if t + h > t_obs:
            break

        # compute p_i
        for i in range(n - s_length):
            p[i] = q_next(theta, s_as_array, not_in_s[i])

        # Draw x ~ Cat(V \ S, (pi)_{i \in V \S })
        sum_p = 0
        random_num = rand() / (1.0 * RAND_MAX) * q_tilde_val
        for i in range(n - s_length):
            sum_p += p[i]
            if random_num <= sum_p:
                chosen_index = i
                break

        else:
            chosen_index = n - s_length - 1

        s.append(not_in_s[chosen_index])
        del not_in_s[chosen_index]

        s_length += 1
        t += h

    free(p)
    state = np.zeros(n, dtype=np.int)

    for i in range(s_length):
        state[s[i]] = 1
    return state


def draw_finite_sample(double[:, :] theta, int sample_size) -> np.ndarray:
    """
    Draw tumor samples from the distribution given by a MHN

    :param theta: matrix containing the theta entries of the MHN
    :param sample_size: number of samples that should be drawn

    :return: returns a binary matrix where rows are samples and columns are genes
    """
    
    cdef int i
    cdef int n = theta.shape[0]
    state_array = np.empty((sample_size, n), dtype=np.int)

    for i in range(sample_size):
        state_array[i, :] = draw_one_sample(theta)


    return state_array
