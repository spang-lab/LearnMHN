"""
This submodule implements UtilityFunctions.R from the original R implementation in Python.

It contains functions useful for preprocessing training data.
"""
# author(s): Stefan Vocht

import numpy as np


def state_to_int(x: np.ndarray) -> int:
    """
    Interprets a binary array as a binary number and returns the corresponding integer value.

    Args:
        x (np.ndarray): Binary array, typically representing a row of the mutation matrix in this context.

    Returns:
        int: The integer value corresponding to the binary array.
    """

    # reverse list and convert elements to string
    x = map(str, x[::-1])
    return int(''.join(x), 2)


def data_to_pD(data: np.ndarray) -> np.ndarray:
    """
    Calculates the probability distribution for the different events from a given binary mutation matrix.

    Args:
        data (np.ndarray): A numpy array or matrix representing the mutation data.

    Returns:
        np.ndarray: The probability distribution of the different events.
    """

    n = data.shape[1]
    N = 2**n

    # convert data into a list of integers, where each number represents a different event
    data = list(map(state_to_int, data))

    # calculate the probability distribution
    pD = np.bincount(data, minlength=N)
    pD = pD / np.sum(pD)

    return pD


def finite_sample(p_th: np.ndarray, k: int) -> np.ndarray:
    """
    Generates a random sample given a probability distribution and returns the probability distribution for the new sample.

    Args:
        p_th (np.ndarray): Probability distribution of events (the distribution of a true Theta).
        k (int): The number of samples to generate.

    Returns:
        np.ndarray: The probability distribution of events from the generated samples.
    """
    n = p_th.size

    return np.bincount(np.random.choice(n, k, replace=True, p=p_th), minlength=n) / k


def KL_div(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the Kullback–Leibler divergence between two probability distributions.

    Args:
        p (np.ndarray): Probability distribution p.
        q (np.ndarray): Probability distribution q.

    Returns:
        float: The KL-divergence of p and q.
    """
    return p.dot(np.log(p)) - p.dot(np.log(q))
