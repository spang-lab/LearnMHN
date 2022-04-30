# by Stefan Vocht
#
# this script implements the UtilityFunctions.R in python

import numpy as np


def state_to_int(x: np.ndarray) -> int:
    """
    This function interprets an binary array x as a binary number and returns the corresponding value as an integer

    :param x: binary array, in the context of the script this is a row of the mutation matrix
    :return: integer number representing the binary array
    """

    # reverse list and convert elements to string
    x = map(str, x[::-1])
    return int(''.join(x), 2)


def data_to_pD(data: np.ndarray) -> np.ndarray:
    """
    This function calculates the probability distribution for the different events for a given binary mutation matrix

    :param data: has to be an numpy array/matrix (mutation matrix)
    :return: probability distribution of the different events
    """

    n = data.shape[1]
    N = 2**n

    # convert data into a list of integers, where each number represents a different event
    data = list(map(state_to_int, data))

    # calculate the probability distribution
    pD = np.bincount(data, minlength=N)
    pD = pD / np.sum(pD)

    return pD


def finite_sample(pTh: np.ndarray, k: int) -> np.ndarray:
    """
    Generates a random sample given a probability distribution and returns the probability distribution for this new
    sample

    :param pTh: probability distribution of events (the distribution of a true Theta)
    :param k: number of samples that should be generated
    :return: probability distribution of events from the generated samples
    """
    n = pTh.size

    return np.bincount(np.random.choice(n, k, replace=True, p=pTh), minlength=n) / k


def KL_div(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the KL-divergence

    :param p: probability distribution p
    :param q: probability distribution q
    :return: KL-divergence of p and q
    """
    return p.dot(np.log(p)) - p.dot(np.log(q))


if __name__ == '__main__':
    print(KL_div(np.array([.1,.3]), np.array([.4,.1])))
