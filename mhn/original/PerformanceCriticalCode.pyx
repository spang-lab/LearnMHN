
# by Stefan Vocht
# 
# this file contains the Cython code equivalent to the original R code in InlineFunctions.R from the original MHN repo
#

cimport cython

from scipy.linalg.cython_blas cimport dcopy, dscal, daxpy, ddot
from libc.stdlib cimport malloc, free
from libc.math cimport exp


cpdef void kron_vec(double[:, :] theta_mat, int i, double[:] x_vec, bint diag = False, bint transp = False):
    """
    This function multiplies the kronecker-product you get from the ith row of theta with a vector

    :param theta: matrix containing the theta values
    :param i: row of theta used for the kronecker-product
    :param x_vec: vector that is multiplied with the kronecker-product matrix
    :param diag: if False, the diagonal of the kronecker-product matrix is set to zero
    :param transp: if True, the kronecker-product matrix is transposed
    :return:
    """
    # inizialize some constants used in this function
    cdef double *theta_i = &theta_mat[i, :]
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
        shuffled_vec = pout
    else:
        swap_vec = pout
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

        old_vec = shuffled_vec;
        shuffled_vec = swap_vec;
        swap_vec = old_vec;

    free(ptmp)


cdef void loop_j(int i, int n, double *pr, double *pG):
    """
    This function is used in the gradient function (in Likelihood.pyx) to compute the gradient more efficiently

    :param i: current row of the gradient to be computed
    :param n: number of columns/rows of theta
    :param r_vec: a vector calculated in the gradient function
    :param g: gradient matrix (output)
    :return:
    """   

    cdef int nx = 1 << n
    cdef int nxhalf = nx/2

    pG = pG + i*n

    cdef double one = 1.
    cdef double *ptmp = <double *> malloc(nx*sizeof(double))

    # instead of copying the shuffled vector back into the original
    # we change the following pointers in such a way that we do not have to copy the data back
    cdef double *shuffled_vec, *old_vec, *swap_vec
    cdef int incx = 1
    cdef int incx2 = 2
    cdef int inc0 = 0
    cdef int j

    old_vec = pr
    shuffled_vec = ptmp

    for j in range(n):
        # matrix shuffle
        dcopy(&nxhalf, old_vec, &incx2, shuffled_vec, &incx)
        dcopy(&nxhalf, old_vec+1, &incx2, shuffled_vec+nxhalf, &incx)

        # sums as dot products with 1
        pG[j] = ddot(&nxhalf, shuffled_vec+nxhalf, &incx, &one, &incx0)
        if i == j:
            pG[j] += ddot(&nxhalf, shuffled_vec, &incx, &one, &incx0)

        swap_vec = old_vec
        old_vec = shuffled_vec
        shuffled_vec = swap_vec

    free(ptmp)
