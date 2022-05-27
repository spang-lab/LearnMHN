# by Stefan Vocht
#
# this script implements Likelihood.R in Cython
#

from . cimport ModelConstruction
from .PerformanceCriticalCode cimport kron_vec, loop_j

cpdef q_vec(theta: np.ndarray, x: np.ndarray, diag: bool = False, transp: bool = False) -> np.ndarray:
    """
    Multiplies the vector x with the matrix Q

    :param theta: thetas used to construct Q
    :param x: vector that is multiplied with Q
    :param diag: if False, the diagonal of Q is set to zero
    :param transp: if True, x is multiplied with Q^T

    :return: product of Q and x
    """
    n = theta.shape[1]
    y = np.zeros(2**n)

    for i in prange(n):
        y += kron_vec(theta, i, x, diag, transp)

    return y