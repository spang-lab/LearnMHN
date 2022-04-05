# by Stefan Vocht
#
# this script contains performance-critical functions (mainly kron_vec and loop_j)
#
# Those functions are also implemented in C and this script will look for the corresponding shared object file (.so)
# and use its functions if possible. Else, it will use code written in python and boosted by numba.
#

import numpy as np
from numba import njit
import os

if os.path.isdir("C") and "libInlineFunctionForPython.so" in os.listdir("C"):
    from cffi import FFI
    print("Use C Lib")
    ffi = FFI()
    ffi.cdef("""void c_kronvec(double *ptheta, int i, double *px, bool diag, bool transp, int n, double *pout);
                void c_loop_j(int i, int n, double* pr, double *pG);""")
    C = ffi.dlopen("C/libInlineFunctionForPython.so")
    c_kronvec = C.c_kronvec
    c_loop_j = C.c_loop_j

    @njit
    def c_kronvec_wrapper(theta: np.ndarray, i: int, x_vec: np.ndarray, diag: bool = False,
                          transp: bool = False) -> np.ndarray:
        """
        This function is a wrapper for the c_kronvec function implemented in C and  takes the same arguments
        as the python version of kron_vec
        """
        n = theta.shape[0]
        result = np.empty((2 ** n))
        c_kronvec(ffi.from_buffer(theta), i, ffi.from_buffer(x_vec), diag, transp, n, ffi.from_buffer(result))
        return result

    @njit
    def c_loop_j_wrapper(i: int, n: int, r_vec: np.ndarray, g: np.ndarray):
        """
        This function is a wrapper for the c_loop_j function implemented in C and takes the same arguments
        as the python version of loop_j
        """
        c_loop_j(i, n, ffi.from_buffer(r_vec), ffi.from_buffer(g))

    kron_vec = c_kronvec_wrapper
    loop_j = c_loop_j_wrapper

else:
    @njit(cache=True)
    def kron_vec(theta: np.ndarray, i: int, x_vec: np.ndarray, diag: bool = False,
                      transp: bool = False) -> np.ndarray:
        """
        This function multiplies the kronecker-product you get from the ith row of theta with a vector

        :param theta: matrix containing the theta values
        :param i: row of theta used for the kronecker-product
        :param x_vec: vector that is multiplied with the kronecker-product matrix
        :param diag: if False, the diagonal of the kronecker-product matrix is set to zero
        :param transp: if True, the kronecker-product matrix is transposed
        :return:
        """
        theta_i = np.exp(theta[i])
        n = theta_i.size

        x_vec = x_vec.astype(np.float64)

        for j in range(n - 1, -1, -1):
            x = x_vec.reshape((2, 2 ** (n - 1)))

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

        return x_vec

    @njit
    def loop_j(i: int, n: int, r_vec: np.ndarray, g: np.ndarray):
        """
        This function is used in the gradient function (in Likelihood.py) to compute the gradient more efficiently

        :param i: current row of the gradient to be computed
        :param n: number of columns/rows of theta
        :param r_vec: a vector calculated in the gradient function
        :param g: gradient matrix (output)
        :return:
        """
        for j in range(n):
            r = r_vec.reshape((2**(n-1), 2))

            g[i, j] += np.sum(r[:, 1])

            if i == j:
                g[i, j] += np.sum(r[:, 0])

            r_vec = r.T.flatten()
