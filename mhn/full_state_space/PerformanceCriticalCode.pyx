"""
This submodule contains the Cython code equivalent to the original R code in InlineFunctions.R from the original MHN repo
as well as some functions to solve linear equations involving [I-Q].
"""
# author(s): Stefan Vocht

cimport cython
import numpy as np
cimport numpy as np

from scipy.linalg.cython_blas cimport dcopy, dscal, daxpy, ddot
from libc.stdlib cimport malloc, free
from libc.math cimport exp

np.import_array()

cdef void internal_kron_vec(double[:, :] theta_mat, int i, double[:] x_vec, double *pout, bint diag, bint transp):
    """
    This function multiplies the kronecker-product you get from the ith row of theta with a vector

    :param theta_mat: matrix containing the theta values
    :param i: row of theta used for the kronecker-product
    :param x_vec: vector that is multiplied with the kronecker-product matrix
    :param pout: vector that will contain the result of the multiplication
    :param diag: if False, the diagonal of the kronecker-product matrix is set to zero
    :param transp: if True, the kronecker-product matrix is transposed
    :return:
    """
    # inizialize some constants used in this function
    cdef double[:] theta_i = theta_mat[i, :]
    cdef int n = theta_mat.shape[0]
    cdef int nx = 1 << n
    cdef int nxhalf = nx / 2
    cdef double mOne = -1
    cdef double zero = 0

    cdef int incx = 1
    cdef int incx2 = 2
    cdef int j

    cdef double *ptmp = <double *> malloc(nx * sizeof(double))
    cdef double *px1
    cdef double *px2
    cdef double *shuffled_vec
    cdef double *old_vec
    cdef double *swap_vec
    cdef double theta

    # for the shuffle algorithm we have to initialize the pointers correctly
    if n & 1 == 1:
        swap_vec = ptmp
        shuffled_vec = &pout[0]
    else:
        swap_vec = &pout[0]
        shuffled_vec = ptmp

    old_vec = &x_vec[0]

    # use the shuffle algorithm to compute the product of the kronecker product with a vector
    for j in range(n):
        dcopy(&nxhalf, old_vec, &incx2, shuffled_vec, &incx)
        dcopy(&nxhalf, old_vec+1, &incx2, shuffled_vec+nxhalf, &incx)

        theta = exp(theta_i[j])
        px1 = shuffled_vec
        px2 = shuffled_vec + nxhalf

        if j == i:
            if not transp:
                dcopy(&nxhalf, px1, &incx, px2, &incx)
                dscal(&nxhalf, &theta, px2, &incx)
                if diag:
                    dcopy(&nxhalf, px2, &incx, px1, &incx)
                    dscal(&nxhalf, &mOne, px1, &incx)
                else:
                    dscal(&nxhalf, &zero, px1, &incx)
            else:
                if diag:
                    theta *= -1
                    daxpy(&nxhalf, &mOne, px2, &incx, px1, &incx)
                    dscal(&nxhalf, &theta, px1, &incx)
                    dscal(&nxhalf, &zero, px2, &incx)
                else:
                    dcopy(&nxhalf,px2,&incx,px1,&incx)
                    dscal(&nxhalf,&theta,px1,&incx)
                    dscal(&nxhalf,&zero,px2,&incx)

        else:
            dscal(&nxhalf, &theta, px2, &incx)

        old_vec = shuffled_vec
        shuffled_vec = swap_vec
        swap_vec = old_vec

    free(ptmp)


def kron_vec(double[:, :] theta_mat, int i, double[:] x_vec, bint diag = False, bint transp = False) -> np.ndarray:
    """
    This function multiplies the kronecker-product you get from the ith row of theta with a vector.
    This is a Python wrapper for the more efficient Cython implementation.

    :param theta_mat: matrix containing the theta values
    :param i: row of theta used for the kronecker-product
    :param x_vec: vector that is multiplied with the kronecker-product matrix
    :param diag: if False, the diagonal of the kronecker-product matrix is set to zero
    :param transp: if True, the kronecker-product matrix is transposed
    :return: vector that will contain the result of the multiplication
    """
    cdef int n = theta_mat.shape[0]
    cdef np.ndarray[np.double_t] result = np.empty(2**n, dtype=np.double)
    internal_kron_vec(theta_mat, i, x_vec, &result[0], diag, transp)
    return result


cdef void loop_j(int i, int n, double *pr, double *pg):
    """
    This function is used in the gradient function (in Likelihood.pyx) to compute the gradient more efficiently

    :param i: current row of the gradient to be computed
    :param n: number of columns/rows of theta
    :param pr: a vector calculated in the gradient function
    :param pg: gradient matrix (output)
    :return:
    """   

    cdef int nx = 1 << n
    cdef int nxhalf = nx/2

    pg = pg + i * n

    cdef double one = 1.
    cdef double *ptmp = <double *> malloc(nx*sizeof(double))

    # instead of copying the shuffled vector back into the original
    # we change the following pointers in such a way that we do not have to copy the data back
    cdef double *shuffled_vec, *old_vec, *swap_vec
    cdef int incx = 1
    cdef int incx2 = 2
    cdef int incx0 = 0
    cdef int j

    old_vec = pr
    shuffled_vec = ptmp

    for j in range(n):
        # matrix shuffle
        dcopy(&nxhalf, old_vec, &incx2, shuffled_vec, &incx)
        dcopy(&nxhalf, old_vec+1, &incx2, shuffled_vec+nxhalf, &incx)

        # sums as dot products with 1
        pg[j] = ddot(&nxhalf, shuffled_vec + nxhalf, &incx, &one, &incx0)
        if i == j:
            pg[j] += ddot(&nxhalf, shuffled_vec, &incx, &one, &incx0)

        swap_vec = old_vec
        old_vec = shuffled_vec
        shuffled_vec = swap_vec

    free(ptmp)


@cython.cdivision(True)
cdef void _compute_inverse(const double *theta, int n, const double *dg, const double *b, double *xout):
    """
    Internal function to compute the solution for [I-Q] x = b using forward substitution
    
    :param theta: thetas used to construct Q
    :param dg: vector containing the diagonal values of [I-Q]
    :param b: a double vector 
    :param xout: solution x as double vector
    """
    cdef int nx = 1 << n
    cdef int incx = 1

    cdef int i, j, k
    cdef int i_copy
    cdef int bit_setter
    cdef int modified_i

    cdef double theta_product
    cdef double xout_i

    # we need the exponential form of theta, so we compute it here once
    cdef double *exp_theta = <double *> malloc(n*n * sizeof(double))
    for i in range(n):
        for j in range(n):
            exp_theta[i*n + j] = exp(theta[i*n + j])

    # we compute the values of xout using forward substitution
    # at the beginning we initialize xout with the values of b
    # in each iteration i we compute the next final value of xout by dividing the current xout value by the ith diagonal entry
    # we then use that final value to update the values of all xout entries that represent states with exactly one more
    # mutation than the state represented by the ith xout entry
    # each index of those other xout entries can be obtained by flipping a bit in i from zero to one
    # if the jth bit is then set to one, this corresponds to the jth gene being mutated, so we have to add the product of
    # theta_jj with all theta_jk, where the kth gene is mutated in i, and with xout[i] to the temporary value of the
    # xout entry that we want to update
    dcopy(&nx, b, &incx, xout, &incx)
    for i in range(nx):
        # get the final value for xout by dividing with the diagonal entry
        xout[i] /= dg[i]
        # cache the value for later
        xout_i = xout[i]

        # update the other xout entries that differ by exactly one bit flipped to one
        # we use bit_setter to flip a zero at the jth position to become a one
        bit_setter = 1
        for j in range(n):
            modified_i = (i | bit_setter)
            if modified_i != i:
                # compute the product of all thetas which correspond to genes that are mutated in i
                theta_product = 1.
                i_copy = i
                for k in range(n):
                    if i_copy & 1:
                        theta_product *= exp_theta[j*n + k]
                    i_copy >>= 1
                xout[modified_i] += theta_product * exp_theta[j*n + j] * xout_i
            bit_setter <<= 1
    free(exp_theta)


@cython.cdivision(True)
cdef void _compute_inverse_t(const double *theta, int n, const double *dg, const double *b, double *xout):
    """
    Internal function to compute the solution for [I-Q]^T x = b using backward substitution

    :param theta: thetas used to construct Q
    :param dg: vector containing the diagonal values of [I-Q]
    :param b: a double vector 
    :param xout: solution x as double vector
    """
    cdef int nx = 1 << n
    cdef int incx = 1

    cdef int i, j, k
    cdef int i_copy
    cdef int bit_setter
    cdef int modified_i

    cdef double theta_product
    cdef double xout_i

    # we need the exponential form of theta, so we compute it here once
    cdef double *exp_theta = <double *> malloc(n * n * sizeof(double))
    for i in range(n):
        for j in range(n):
            exp_theta[i*n + j] = exp(theta[i*n + j])

    # initialize xout with the values of b
    dcopy(&nx, b, &incx, xout, &incx)

    # we compute the values of xout using backward substitution
    # at the beginning we initialize xout with the values of b
    # in each iteration i we compute the next final value of xout by dividing the current xout value by the ith diagonal entry
    # we then use that final value to update the values of all xout entries that represent states with exactly one less
    # mutation than the state represented by the ith xout entry
    # each index of those other xout entries can be obtained by flipping a bit in i from one to zero
    # if the jth bit is then set to zero, this corresponds to reversing the mutation of the jth gene, so we have to add the product of
    # theta_jj with all theta_jk, where the kth gene is mutated in i, and with xout[i] to the temporary value of the
    # xout entry that we want to update
    for i in range(nx-1, -1, -1):
        # get the final value for xout by dividing with the diagonal entry
        xout[i] /= dg[i]
        # cache the value for later
        xout_i = xout[i]
        # update the other xout entries that differ by exactly one bit flipped to zero
        # we use bit_setter to flip a one at the jth position to become a zero
        bit_setter = 1
        for j in range(n):
            modified_i = (i & (~bit_setter))
            if modified_i != i:
                # compute the product of all thetas which correspond to genes that are mutated in i
                theta_product = 1.
                i_copy = modified_i
                for k in range(n):
                    if i_copy & 1:
                        theta_product *= exp_theta[j*n + k]
                    i_copy >>= 1
                xout[modified_i] += theta_product * exp_theta[j*n + j] * xout_i
            bit_setter <<= 1
    free(exp_theta)


cpdef compute_inverse(double[:, :] theta, double[:] dg, double[:] b, double[:] xout, bint transp):
    """
    Computes the solution for [I - Q] x = b using forward (and backward) substitution.

    :param theta: thetas used to construct Q
    :param dg: vector containing the diagonal values of [I-Q]
    :param b: a double vector
    :param xout: double vector that will contain the solution after running this function
    :param transp: if True, returns solution for [I - Q]^T x = b
    """
    cdef int n = theta.shape[0]
    if transp:
        _compute_inverse_t(&theta[0, 0], n, &dg[0], &b[0], &xout[0])
    else:
        _compute_inverse(&theta[0, 0], n, &dg[0], &b[0], &xout[0])