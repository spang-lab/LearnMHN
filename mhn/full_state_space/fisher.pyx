"""
This submodule implements the Fisher information matrix for cMHN and oMHN.

It contains function for computing the FIM in Cython and for calling the
CUDA implementation.
"""

# authors: Stefan Vocht, Y. Linda Hu

from scipy.linalg.cython_blas cimport dcopy, dscal, daxpy, ddot
from libc.stdlib cimport malloc

import numpy as np
cimport numpy as np

from .PerformanceCriticalCode cimport _compute_inverse_t, _compute_inverse,\
                                      internal_kron_vec
from .ModelConstruction import q_diag
from .Likelihood import cuda_fisher

np.import_array()

def cython_fisher(
    double[:, :] theta
):
    """
    Computes the fisher information matrix for cMHN.
    The formulas are described in S. Vocht. Identifiability of
    Mutual Hazard Networks. Unpublished bachelor thesis, 2022

    Args:
        theta (np.ndarray): matrix containing the theta values

    Returns:
        np.ndarray: The Fisher information matrix for theta
    """
    cdef int i, j, s, t
    cdef int incx = 1
    cdef int incx2 = 2
    cdef int incx0 = 0
    cdef double zero = 0.
    cdef double one = 1.
    cdef double mOne = -1.
    cdef int n = theta.shape[0]
    cdef int nx = 1 << n
    cdef int nxhalf = nx // 2

    cdef double *shuffled_vec = <double *> malloc(nx * sizeof(double))
    cdef double *q_st = <double *> malloc(nx * sizeof(double))
    cdef double *dQ_st = <double *> malloc(nx * sizeof(double))
    cdef double *masked_dQ_st = <double *> malloc(nx * sizeof(double))
    cdef double *zero_mask = <double *> malloc(nx * sizeof(double))
    cdef double *dQ_ij = <double *> malloc(nx * sizeof(double))
    cdef double *swap_vec
    cdef np.ndarray[np.double_t] pth_ = np.empty(nx, dtype=np.double)

    cdef np.ndarray[np.double_t, ndim=2] fisher_mat = np.zeros((n * n, n * n))

    # get p_th
    cdef np.ndarray[np.double_t] p0 = np.zeros(nx, dtype=np.double)
    p0[0] = 1
    # get the diagonal of I-Q
    cdef np.ndarray[np.double_t, ndim=1] dg = q_diag(theta)
    daxpy(&nx, &one, &mOne, &incx0, &dg[0], &incx)
    dscal(&nx, &mOne, &dg[0], &incx)
    _compute_inverse(&theta[0,0], n, &dg[0], &p0[0], &pth_[0])

    ind_x = -1

    for s in range(n):
        # kron_vec function from the original MHN implementation
        internal_kron_vec(
            theta, s, pth_, dQ_st, 1, 0
        )
        # set up zero_mask
        dcopy(&nxhalf, &zero, &incx0, zero_mask, &incx2)
        dcopy(&nxhalf, &one, &incx0, zero_mask + 1, &incx2)

        for t in range(n):
            # the derivative of Q wrt. theta_st ( here stored in
            # masked_dQ_st) mainly depends on s
            # the difference between the theta_st is only that for
            # s != t some entries are set to zero
            # this is done here using zero_mask
            
            if s != t:
                for i in range(nx):
                    masked_dQ_st[i] = dQ_st[i] * zero_mask[i]
            else:
                dcopy(&nx, dQ_st, &incx, masked_dQ_st, &incx)

            
            # shuffle zero mask
            dcopy(&nxhalf, zero_mask, &incx, shuffled_vec, &incx2)
            dcopy(&nxhalf, zero_mask + nxhalf, &incx, shuffled_vec+1, &incx2)

            swap_vec = shuffled_vec
            shuffled_vec = zero_mask
            zero_mask = swap_vec
            # q_st described in section 4.1.1
            # compute [I-Q]^( -1) * masked_dQ_st
            _compute_inverse(&theta[0,0], n, &dg[0], masked_dQ_st, q_st)
            
            for i in range(nx):
                q_st[i] /= pth_[i]
            
            _compute_inverse_t(&theta[0,0], n, &dg[0], q_st, shuffled_vec)
            dcopy(&nx, shuffled_vec, &incx, q_st, &incx)
            ind_x += 1
            ind_y = s * n
            # fisher matrix is symmetric , compute only the upper half
            # and use the shuffle trick for more efficiency
            for i in range(s, n):

                internal_kron_vec(theta, i, pth_, dQ_ij, 1, 0)

                for k in range(nx):
                    dQ_ij[k] *= q_st[k]

                for j in range(n):
                    
                    # reshaping for shuffling
                    dcopy(&nxhalf, dQ_ij, &incx2, shuffled_vec, &incx)
                    dcopy(&nxhalf, dQ_ij+1, &incx2, shuffled_vec+nxhalf, &incx)
                    
                    # compute only the upper half
                    if ind_y >= ind_x:

                        fisher_mat[ind_x, ind_y] = ddot(
                            &nxhalf, shuffled_vec+nxhalf, &incx, &one, &incx0)
                
                        if i == j:
                            fisher_mat[ind_x, ind_y] += ddot(
                                &nxhalf, shuffled_vec, &incx, &one, &incx0)
                    
                    ind_y += 1

                    # swap dQ_ij and shuffled_vec
                    swap_vec = dQ_ij
                    dQ_ij = shuffled_vec
                    shuffled_vec = swap_vec
 
    # fill the lower triangular part
    fisher_mat += np.tril(fisher_mat.T, -1)
    return fisher_mat

def omhn_fisher(
    log_theta: np.ndarray,
    use_cuda: bool = False,
) -> np.ndarray:
    """
    Computes the Fisher information matrix for oMHN.
    
    Args:
        log_theta (np.ndarray): matrix containing the log theta values
        use_cuda (bool, optional): whether to use GPU acceleration
    
    Returns:
        np.ndarray: The Fisher information matrix for log_theta
    """

    n = log_theta.shape[1]

    # subtract observation rates from each element in each column
    cmhn_log_theta = log_theta[:-1] - log_theta[-1]
    
    # undo changes to the diagonal
    cmhn_log_theta[np.diag_indices(n)] += log_theta[-1]

    #        / theta_theta | omega_theta \
    # FIM = |--------------+--------------|
    #        \ omega_theta | omega_omega /

    theta_theta = fisher(
        cmhn_log_theta, omhn=False, use_cuda=use_cuda)

    # F_{\omega_j}{\theta_st} = \sum_{i|=j} F_{\theta_ij}{\theta_st}
    theta_theta_reshape = theta_theta.reshape(n, n, n**2, order="F")
    omega_theta = -(
        theta_theta_reshape.sum(axis=1)
        - theta_theta_reshape[np.diag_indices(n)]
    )

    omega_theta_reshape = omega_theta.reshape(n, n, n, order="F")
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    omega_omega = -(
        omega_theta_reshape.sum(axis=2) - omega_theta_reshape[i, j, j]
    )

    return np.block([[theta_theta, omega_theta.T], [omega_theta, omega_omega]])


def fisher(
    log_theta: np.ndarray, omhn: bool = True, use_cuda: bool = False
) -> np.ndarray:
    """
    Computes the Fisher information matrix for o/cMHN.
    
    Args:
        log_theta(np.ndarray): matrix containing the log theta values
        omhn (bool, optional): whether MHN is oMHN or cMHN
        use_cuda (bool, optional): whether to use GPU acceleration
    
    Returns:
        np.ndarray: The Fisher information matrix for log_theta
    """
    if omhn:
        return omhn_fisher(log_theta, use_cuda=use_cuda)
    else:
        if use_cuda:
            return cuda_fisher(log_theta)
        else:
            return cython_fisher(log_theta)