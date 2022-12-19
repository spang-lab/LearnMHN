# by Stefan Vocht
#
# this script implements the functions of ModelConstruction.R in Cython
#

cimport cython

from libc.math cimport exp

from scipy.sparse import diags as scipy_diags

import numpy as np
cimport numpy as np

np.import_array()

def random_theta(n: int, sparsity: float = 0, rounded: bool = True) -> np.ndarray:
    """
    Creates a random MHN with (log-transformed) parameters Theta

    :param n: size of Theta -> size of corresponding Q is 2^n
    :param sparsity: sparsity of Theta as percentage
    :param rounded: if True, the random Theta is rounded to two decimals
    :return: random Theta
    """
    theta = np.zeros((n, n))

    np.fill_diagonal(theta, -1)
    theta = theta.flatten()
    non_zeros = np.random.choice(np.argwhere(theta != -1).squeeze(), size=int((n**2 - n)*(1 - sparsity)))

    theta[non_zeros] = np.random.normal(size=non_zeros.size)
    theta = theta.reshape((n, n))
    np.fill_diagonal(theta, np.random.normal(size=n))

    if rounded:
        theta = np.around(theta, decimals=2)

    return theta


cpdef np.ndarray[np.double_t, ndim=1] q_subdiag(double[:, :] theta, int i):
    """
    Creates a single subdiagonal of Q from the ith row in Theta

    :return: subdiagonal of Q corresponding to the ith row of Theta
    """
    cdef double[:] row = theta[i, :]
    cdef int n = theta.shape[0]
    cdef int j

    # s is the subdiagonal of Q, the entries are calculated as described in eq. 2
    cdef np.ndarray[np.double_t, ndim=1] s = np.empty(2**n, dtype=np.double)
    s[0] = exp(row[i])

    for j in range(n):
        s[2**j: 2**(j+1)] = s[:2**j] * exp(row[j]) * (i != j)

    return s


def build_q(theta: np.ndarray) -> np.ndarray:
    """
    Build the transition rate matrix Q from its subdiagonals for a given Theta

    :param theta: matrix representing the MHN
    :return: rate matrix Q
    """
    n = theta.shape[0]

    subdiags = np.array(list(map(lambda i: q_subdiag(theta, i), range(n))))
    diag = -np.sum(subdiags, axis=0)

    # scipy's diag cuts the end of the subdiags that dont fit into the final matrix Q
    q = scipy_diags(np.vstack((diag, subdiags)), offsets=[0] + [-2**i for i in range(n)], shape=(2**n, 2**n))

    return q.toarray()


cpdef np.ndarray[np.double_t, ndim=1]  q_diag(double[:, :] theta):
    """
    get the diagonal of Q

    :param theta: theta representing the MHN
    """
    cdef int n = theta.shape[0]
    cdef int i
    cdef np.ndarray[np.double_t, ndim=1] dg = np.zeros((2**n), dtype=np.double)

    for i in range(n):
        # the diagonal elements are the negative sums of their columns
        dg = dg - q_subdiag(theta, i)

    return dg


def learn_indep(pD: np.ndarray) -> np.ndarray:
    """
    Learns an independence model from the data distribution, which assumes that no events interact.
    Used to initialize the parameters of the actual model before optimization

    :param pD: probability distribution of the events in the data
    :return: independence model
    """
    cdef int n = int(np.log2(pD.size))
    cdef int i
    cdef np.ndarray[np.double_t, ndim=2] theta = np.zeros((n, n), dtype=np.double)

    # for each event i, sum up the probabilities of events where gene i was mutated to get the total probability
    # theta_ii is the log-odd of gene i being mutated
    for i in range(n):
        pD = pD.T.flatten()
        pD = pD.reshape((2**(n-1), 2))

        perc = np.sum(pD[:, 1])

        # fill theta_ii with the log-odds of gene i being mutated in a sample
        theta[i, i] = np.log(perc / (1 - perc))

    return np.around(theta, decimals=2)