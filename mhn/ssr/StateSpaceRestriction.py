# by Stefan Vocht
#
# this file contains the functions necessary to train a MHN using state space restriction
#

import numpy as np

from numpy.linalg import inv

from MHN import ModelConstruction, UtilityFunctions
from MHN import Likelihood


def count_ones(u: int) -> int:
    """
    Counts number of 1s in binary representation of number u
    Source:
    https://web.archive.org/web/20151229003112/http://blogs.msdn.com/b/jeuge/archive/2005/06/08/hakmem-bit-count.aspx
    and https://stackoverflow.com/questions/8871204/count-number-of-1s-in-binary-representation
    """
    count = u - ((u >> 1) & 0o033333333333) - ((u >> 2) & 0o011111111111)
    return ((count + (count >> 3)) & 0o030707070707) % 63


def restricted_kronvec(theta: np.ndarray, i: int, x_vec: np.ndarray, state: int, diag: bool = False, transp: bool = False) -> np.ndarray:
    """
    This function multiplies the kronecker product described in the original MHN paper in eq.9 with a vector

    :param theta: matrix containing the theta entries
    :param i: vector is multiplied with the ith kronecker product (ith summand in eq. 9 of the original paper)
    :param x_vec: vector that is multiplied with the kronecker product
    :param state: current state used to compute the gradient
    :param diag: if False, the diagonal of the kronecker product is set to zero
    :param transp: if True, the kronecker product is transposed
    """

    # if we have no diagonal and the ith gene is not mutated, the result is always a zero vector
    if not diag and not (state >> i) & 1:
        return np.zeros(x_vec.size)

    theta_i = np.exp(theta[i])
    n = theta_i.size
    mutation_num = count_ones(state)

    x_vec = x_vec.astype(np.float64)

    comparator = 1 << (n - 1)

    # use the shuffle algorithm to compute the product of the kronecker product with a vector
    for j in range(n-1, -1, -1):
        if state & comparator:
            x = x_vec.reshape(2, 2**(mutation_num-1))
            if j == i:
                if not transp:
                    x[1] = x[0] * theta_i[j]
                    if diag:
                        x[0] = -x[1]
                    else:
                        x[0] = 0
                else:
                    if diag:
                        x[0] = (x[1] - x[0]) * theta_i[j]
                        x[1] = 0
                    else:
                        x[0] = x[1] * theta_i[j]
                        x[1] = 0

            else:
                x[1] = x[1] * theta_i[j]

            x_vec = x.T.flatten()
        elif i == j:
            x_vec *= - theta_i[j]

        state <<= 1

    return x_vec


def restricted_q_vec(theta: np.ndarray, x: np.ndarray, state: int, diag: bool = False, transp: bool = False) -> np.ndarray:
    """
    computes y = Q(ptheta) * x, result is saved in yout

    :param theta: matrix containing the theta entries
    :param x: vector that should be multiplied with Q(ptheta)
    :param state: state representing current tumor sample
    :param diag: if False, the diag of Q is set to zero during multiplication
    :param transp: if True, multiplication is done with the transposed Q

    :returns: y
    """
    n = theta.shape[0]
    y = np.zeros(x.size)

    for i in range(n):
        y += restricted_kronvec(theta, i, x, state, diag, transp)

    return y


def restricted_q_diag(theta: np.ndarray, state: int) -> np.ndarray:
    """
    Compute the diagonal of the transition rate matrix Q

    :param theta: matrix containing the theta entries
    :param state: state representing current tumor sample

    :returns: the diagonal of Q
    """
    mutation_num = count_ones(state)
    dg = np.zeros(2**mutation_num)
    n = theta.shape[0]

    for i in range(n):
        state_copy = state
        s = np.ones(1)
        for j in range(n):
            if state_copy & 1:
                if i == j:
                    s = np.append(-s * np.exp(theta[i, j]), s * 0)
                else:
                    s = np.append(s, s * np.exp(theta[i, j]))

            elif i == j:
                s *= - np.exp(theta[i, j])
            state_copy >>= 1
        dg += s

    return dg


def restricted_jacobi(theta: np.ndarray, b: np.ndarray, state: int, transp: bool = False) -> np.ndarray:
    """
    this functions multiplies [I-Q]^(-1) with b

    :param theta: matrix containing the theta entries
    :param b: array that is multiplied with [I-Q]^(-1)
    :param state: state representing current tumor sample
    :param transp: if True, b is multiplied with the tranposed [I-Q]^(-1)
    """
    mutation_num = count_ones(state)

    x = np.full(b.size, 1 / b.size)

    # compute the diagonal of [I-Q]
    dg = 1 - restricted_q_diag(theta, state)

    for _ in range(mutation_num+1):
        x = b + restricted_q_vec(theta, x, state, diag=False, transp=transp)
        x = x / dg

    return x


def restricted_gradient(theta: np.ndarray, state: int) -> np.ndarray:
    """
    Computes a part of the gradient corresponding to a given state

    :param theta: matrix containing the theta entries
    :param state: state representing current tumor sample

    :returns: the gradient corresponding to the given state
    """
    n = theta.shape[0]
    mutation_num = count_ones(state)
    p0 = np.zeros(2**mutation_num)
    p0[0] = 1

    # compute parts of the probability distribution yielded by the current MHN
    pth = restricted_jacobi(theta, p0, state)

    pD = np.zeros(2**mutation_num)
    pD[-1] = 1 / pth[-1]

    q = restricted_jacobi(theta, pD, state, transp=True)
    g = np.zeros((n, n))

    # compute the gradient efficiently using the shuffle trick
    for i in range(n):
        r_vec = q * restricted_kronvec(theta, i, pth, state, diag=True)
        state_copy = state
        for j in range(n):
            if state_copy & 1:
                r = r_vec.reshape(2**(mutation_num-1), 2)
                g[i, j] = np.sum(r[:, 1])
                if i == j:
                    g[i, j] += np.sum(r[:, 0])

                r_vec = r.T.flatten()
            elif i == j:
                g[i, j] = np.sum(r_vec)
            state_copy >>= 1

    return g


def gradient(theta: np.ndarray, mutation_data: list) -> np.ndarray:
    """
    Computes the total gradient for a given MHN and given mutation data

    :param theta: matrix containing the theta entries of the current MHN
    :param mutation_data: list containing the mutation data the MHN should be trained on
    """
    n = theta.shape[0]
    final_gradient = np.zeros((n, n))

    for data_point in mutation_data:
        final_gradient += restricted_gradient(theta, data_point)

    return final_gradient / len(mutation_data)


def test():
    """
    Test the implementation of the State Space Restriction
    :return:
    """
    n = 16
    # create a random theta and mutation data
    theta = ModelConstruction.random_theta(n)
    raw_data = np.random.choice([0, 1], size=(124, n), replace=True, p=(0.9, .1))
    mutation_data = []
    for row in raw_data:
        mutation_data.append(UtilityFunctions.state_to_int(row))

    # convert the raw data into a probability distribution
    pD = UtilityFunctions.data_to_pD(raw_data)
    print(mutation_data)
    print({i: pD[i] for i in range(2**n)})

    import time

    # measure runtime of original gradient compared with State Space Restriction
    # and compare the resulting gradients (should be the same)
    t = time.time()
    g1 = gradient(theta, mutation_data)
    print(time.time() - t)
    t = time.time()
    g2 = Likelihood.grad(theta, pD)
    print(time.time() - t)
    print(g1)
    print(g2)
    print(g1 / g2)


if __name__ == '__main__':
    test()
