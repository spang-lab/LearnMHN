"""
This module contains some utility functions for working with MHNs.
"""
# author(s): Stefan Vocht

cimport cython
from libc.math cimport exp

import numpy as np
cimport numpy as np


np.import_array()


@cython.wraparound(False)
@cython.boundscheck(False)
def sample_artificial_data(np.ndarray[np.double_t, ndim=2] theta, int sample_num) -> np.ndarray:
    """
    Returns artificial cross-sectional data sampled from an cMHN. Random values are generated with numpy,
    use np.random.seed() to make results reproducible.

    :param theta: theta matrix representing an cMHN
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


@cython.wraparound(False)
@cython.boundscheck(False)
def gillespie(np.ndarray[np.double_t, ndim=2] theta, np.ndarray[np.int32_t, ndim=1] initial_state, int sample_num) -> tuple[list[list[int]], np.ndarray]:
    """
    Gillespie algorithm to simulate event accumulation.

    :param theta: theta matrix representing the either a cMHN or an oMHN
    :param initial_state: Array representing the starting state, each entry corresponds to an event being present (1) or not (0)
    :param sample_num: number of samples that should be simulated

    :returns: A tuple: first element is a list of lists containing the active events in the correct time order, second element is a numpy array that contains the observation times

    :raise ValueError: if the size of theta is neither (n, n) nor (n+1, n), a ValueError is raised
    """
    cdef int n = theta.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] exp_theta = np.exp(theta)
    cdef np.ndarray[np.double_t, ndim=1] observation_times = np.empty(sample_num, dtype=np.double)

    cdef list trajectory_list = []
    cdef list trajectory

    cdef np.ndarray[np.int32_t] in_current_sample = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t] possible_gene_mutations = np.empty(n, dtype=np.int32)
    cdef np.ndarray[np.double_t] rates_from_current_state = np.empty(n, dtype=np.double)
    cdef np.ndarray[np.double_t] observation_rates

    if theta.shape[0] == n:
        # cMHN
        observation_rates = np.ones(n, dtype=np.double)
    elif theta.shape[0] == n+1:
        # oMHN
        observation_rates = exp_theta[n]
    else:
        raise ValueError(f"Theta must have dimensions {n}x{n} or {n+1}x{n}, but is has dimension {theta.shape[0]}x{theta.shape[1]}")


    cdef int j, gene, mutated_gene
    cdef int sample_index, current_sample_length
    cdef double current_time, passed_time
    cdef double sum_rates, rate, random_crit, accumulated_rate

    cdef int initial_sample_length = initial_state.sum()

    for sample_index in range(sample_num):
        in_current_sample[:] = initial_state.copy()
        current_sample_length = initial_sample_length
        current_time = 0.
        trajectory = [gene for gene in range(n) if initial_state[gene]]
        while 1:
            sum_rates = 0.
            j = 0

            # compute observation rate first and add it to the total sum of rates that lead out of the current state
            rate = 1.
            for mutated_gene in range(n):
                if in_current_sample[mutated_gene]:
                    rate *= observation_rates[mutated_gene]
            sum_rates += rate

            # now compute all the rates that lead to other states
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

            passed_time = np.random.exponential(1 / sum_rates)
            current_time += passed_time
            random_crit = np.random.random(1)[0] * sum_rates
            accumulated_rate = 0.
            for j in range(n - current_sample_length):
                rate = rates_from_current_state[j]
                gene = possible_gene_mutations[j]
                accumulated_rate += rate
                if random_crit <= accumulated_rate:
                    in_current_sample[gene] = 1
                    current_sample_length += 1
                    trajectory.append(gene)
                    break
            else:
                # if we did not leave the loop early, this means no state was selected as next state
                # instead the observation event happened
                trajectory_list.append(trajectory)
                observation_times[sample_index] = current_time
                break

    return trajectory_list, observation_times


def compute_next_event_probs(np.ndarray[np.double_t, ndim=2] theta, np.ndarray[np.int32_t, ndim=1] current_state, double observation_rate = 0) -> np.ndarray:
    """
    Compute the probability for each event that it will be the next one to occur given the current state.

    :param theta: theta matrix representing an cMHN
    :param current_state: array representing the current state, each entry corresponds to an event being present (1) or not (0)
    :param observation_rate: rate of the observation event, by default set to 0 which means no observation before some other event occurs
    :returns: array that contains the probability for each event that it will be the next one to occur

    :raise ValueError: if the size of theta does not match the size of current_state, a ValueError is raised
    """
    cdef int n = theta.shape[1]  # use shape[1] to be compatible with oMHN
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