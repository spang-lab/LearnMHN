# by Stefan Vocht
#
# this file acts like a C header file for PerformanceCriticalCode.pyx
#


cpdef void kron_vec(double[:, :] theta_mat, int i, double[:] x_vec, double[:] pout, bint diag = False, bint transp = False):
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


cdef void loop_j(int i, int n, double *pr, double *pG):
    """
    This function is used in the gradient function (in Likelihood.pyx) to compute the gradient more efficiently
    
    :param i: current row of the gradient to be computed
    :param n: number of columns/rows of theta
    :param r_vec: a vector calculated in the gradient function
    :param g: gradient matrix (output)
    :return:
    """
