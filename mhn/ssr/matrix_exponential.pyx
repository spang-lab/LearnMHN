"""
This part of the package contains functions related to (differentiated) uniformization as well as functions
to compute the log-likelihood score and its gradient for datasets that contain samples with known ages.

(see Rupp et al.(2021): 'Differentiated uniformization: A new method for inferring Markov chains on combinatorial state spaces including stochastic epidemic models')
"""
# author(s): Kevin Rupp, Stefan Vocht

cimport cython

import warnings

from scipy.linalg.cython_blas cimport dcopy, dscal, daxpy, ddot, dnrm2, dasum
from libc.stdlib cimport malloc, free
from libc.math cimport exp, log

from mhn.ssr.state_containers cimport State, StateContainer, StateAgeContainer
from mhn.ssr.state_space_restriction cimport get_mutation_num, restricted_q_vec, restricted_q_diag

import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

np.import_array()


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void restricted_derivative_ik(double[:, :] theta_mat, int i, double[:] x_vec, State *state, int mutation_num, int k,
                                double *pout) nogil:
    """
    This function multiplies the kronecker product for the partial derivative of Q wrt to theta_ik with a vector

    :param theta_mat: matrix containing the theta entries
    :param i: vector is multiplied with the ith kronecker product (ith summand in eq. 9 of the original paper)
    :param x_vec: vector that is multiplied with the kronecker product
    :param state: current state used to compute the gradient
    :param mutation_num: number of mutations present in state
    :param k: column index of derivative
    :param pout: vector which will contain the result of this multiplication
    """

    # initialize some constants used in this function
    cdef double[:] theta_i = theta_mat[i, :]
    cdef int n = theta_i.shape[0]
    cdef int nx = 1 << mutation_num
    cdef int nxhalf = nx / 2
    cdef double mOne = -1
    cdef double zero = 0

    cdef int incx = 1
    cdef int incx2 = 2
    cdef int j

    # if the kth gene is not mutated and i != k (theta_ik no base rate), then the derivative is always zero
    if i != k and not (state[0].parts[k >> 5] >> (k & 31)) & 1:
        dscal(&nx, &zero, pout, &incx)
        return

    cdef double *ptmp = <double *> malloc(nx * sizeof(double))
    cdef double *px1
    cdef double *px2
    cdef double *shuffled_vec
    cdef double *old_vec
    cdef double *swap_vec
    cdef double theta

    # for the shuffle algorithm we have to initialize the pointers correctly
    if mutation_num & 1 == 1:
        swap_vec = ptmp
        shuffled_vec = pout
    else:
        swap_vec = pout
        shuffled_vec = ptmp

    old_vec = &x_vec[0]

    cdef int state_copy = state[0].parts[0]

    # use the shuffle algorithm to compute the product of the kronecker product with a vector
    for j in range(n):
        if state_copy & 1:
            dcopy(&nxhalf, old_vec, &incx2, shuffled_vec, &incx)
            dcopy(&nxhalf, old_vec+1, &incx2, shuffled_vec+nxhalf, &incx)

            theta = exp(theta_i[j])
            px1 = shuffled_vec
            px2 = shuffled_vec + nxhalf
            # this function is only needed for the dua where we never have to transpose Q' or delete the diagonal
            if j == i:
                dcopy(&nxhalf, px1, &incx, px2, &incx)
                dscal(&nxhalf, &theta, px2, &incx)
                dcopy(&nxhalf, px2, &incx, px1, &incx)
                dscal(&nxhalf, &mOne, px1, &incx)

            else:
                dscal(&nxhalf, &theta, px2, &incx)
                if j == k:
                    dscal(&nxhalf, &zero, px1, &incx)

            old_vec = shuffled_vec
            shuffled_vec = swap_vec
            swap_vec = old_vec

        elif i == j:
            theta = -exp(theta_i[j])

            # if old_vec is still pointing to x_vec, we have to change it to not alter x_vec
            if old_vec == &x_vec[0]:
                dcopy(&nx, old_vec, &incx, swap_vec, &incx)
                old_vec = swap_vec

            dscal(&nx, &theta, old_vec, &incx)

		# if the mutation state of the next gene is stored on the current state_copy, make a bit shift to the right
		# else state_copy becomes the next integer stored in the given state (x >> 5  <=> x // 32, x & 31 <=> x % 32)
        if (j + 1) & 31:
            state_copy >>= 1
        else:
            state_copy = state[0].parts[(j+1) >> 5]

    free(ptmp)

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void restricted_derivative_ik_diag(double[:, :] theta_mat, int i, State *state, int mutation_num, int k,
                                double *pout) except *:
    """
    This function calculates the diagonal of dQ/d theta_ik

    :param theta_mat: matrix containing the theta entries
    :param i: vector is multiplied with the ith kronecker product (ith summand in eq. 9 of the original paper)
    :param state: current state used to compute the gradient
    :param mutation_num: number of mutations present in state
    :param k: column index of derivative
    :param pout: vector which will contain the result of this multiplication
    """

    # initialize some constants used in this function
    cdef double[:] theta_i = theta_mat[i, :]
    cdef int n = theta_i.shape[0]
    cdef int nx = 1 << mutation_num
    cdef int nxhalf = nx / 2
    cdef double mOne = -1
    cdef double zero = 0

    cdef int incx = 1
    cdef int incx2 = 2
    cdef int j

    # if the kth gene is not mutated and i != k (theta_ik no base rate), then the derivative is always zero
    if i != k and not (state[0].parts[k >> 5] >> (k & 31)) & 1:
        dscal(&nx, &zero, pout, &incx)
        return

    cdef double *ptmp = <double *> malloc(nx * sizeof(double))
    cdef double *px1
    cdef double *px2
    cdef double *shuffled_vec
    cdef double *old_vec
    cdef double *swap_vec
    cdef double theta
    cdef double[:] x_vec = np.ones(nx, dtype=np.double)

    # for the shuffle algorithm we have to initialize the pointers correctly
    if mutation_num & 1 == 1:
        swap_vec = ptmp
        shuffled_vec = pout
    else:
        swap_vec = pout
        shuffled_vec = ptmp

    old_vec = &x_vec[0]

    cdef int state_copy = state[0].parts[0]

    # use the shuffle algorithm to compute the product of the kronecker product with a vector
    for j in range(n):
        if state_copy & 1:
            dcopy(&nxhalf, old_vec, &incx2, shuffled_vec, &incx)
            dcopy(&nxhalf, old_vec+1, &incx2, shuffled_vec+nxhalf, &incx)

            theta = exp(theta_i[j])
            px1 = shuffled_vec
            px2 = shuffled_vec + nxhalf
            # this function is only needed for the dua where we never have to transpose dQ or remove its diagonal
            if j == i:
                dcopy(&nxhalf, px1, &incx, px2, &incx)
                dscal(&nxhalf, &theta, px2, &incx)
                dcopy(&nxhalf, px2, &incx, px1, &incx)
                dscal(&nxhalf, &mOne, px1, &incx)
                dscal(&nxhalf, &zero, px2, &incx)

            else:
                dscal(&nxhalf, &theta, px2, &incx)
                if j == k:
                    dscal(&nxhalf, &zero, px1, &incx)

            old_vec = shuffled_vec
            shuffled_vec = swap_vec
            swap_vec = old_vec

        elif i == j:
            theta = -exp(theta_i[j])

            # if old_vec is still pointing to x_vec, we have to change it to not alter x_vec
            if old_vec == &x_vec[0]:
                dcopy(&nx, old_vec, &incx, swap_vec, &incx)
                old_vec = swap_vec

            dscal(&nx, &theta, old_vec, &incx)

		# if the mutation state of the next gene is stored on the current state_copy, make a bit shift to the right
		# else state_copy becomes the next integer stored in the given state (x >> 5  <=> x // 32, x & 31 <=> x % 32)
        if (j + 1) & 31:
            state_copy >>= 1
        else:
            state_copy = state[0].parts[(j+1) >> 5]

    free(ptmp)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef double[:] cython_restricted_expm(double[:, :] theta, double[:] b, State *state, double t, double eps) except *:
    """
    this functions multiplies expm(tQ) with a vector b

    :param theta: matrix containing the theta entries
    :param b: array that is multiplied with expm(tQ)
    :param state: state representing current tumor sample
    :param t: age of state
    :param eps: accuracy
    """
    cdef int mutation_num = get_mutation_num(state)
    cdef int nx = 1 << mutation_num
    cdef int i_one = 1

    # Compute the diagonal of Q
    cdef double *dg = <double *> malloc(nx * sizeof(double))
    restricted_q_diag(theta, state, dg)
    # Calculate the L2-norm of the diagonal to use as scaling constant gamma
    cdef double gam =  dnrm2(&nx, dg, &i_one)
    cdef double gam_inv = 1/gam

    cdef np.ndarray[np.double_t] pt = np.zeros(nx, dtype=np.double)
    cdef int n = 0
    cdef double w = 1.0
    cdef double egtw = exp(-1.0*gam*t)
    cdef double *q_vec_result = <double *> malloc(nx * sizeof(double))
    cdef double[:] q = b.copy()
    cdef double mass_defect = 0.0
    cdef double b_sum = np.sum(b)

    while eps < b_sum * (1 - mass_defect):
        mass_defect += egtw
        daxpy(&nx, &egtw, &q[0], &i_one, &pt[0], &i_one)
        n += 1
        # Calculate q = [1/gamma*Q+I]b
        restricted_q_vec(theta, q, state, q_vec_result, diag=True, transp=False) # q=1/gamma*Qb
        daxpy(&nx, &gam_inv, q_vec_result, &i_one, &q[0], &i_one) # calculate q=q+Ib
        egtw *= gam*t/n

    free(dg)
    free(q_vec_result)
    return pt


@cython.wraparound(False)
@cython.boundscheck(False)
def restricted_expm(double[:, :] theta, double[:] b, StateAgeContainer state_with_age, double eps):
    """
    this functions multiplies expm(tQ) with a vector b, this is a Python wrapper for the internal function cython_restricted_expm

    :param theta: matrix containing the theta entries
    :param b: array that is multiplied with expm(tQ)
    :param state_with_age: StateAgeContainer containing exactly one state with its corresponding age
    :param eps: accuracy
    """
    if state_with_age.data_size != 1:
        raise ValueError("state_with_age is expected to contain exactly one state with its corresponding age")
    cdef State *state = &state_with_age.states[0]
    cdef double t = state_with_age.state_ages[0]
    return np.asarray(cython_restricted_expm(theta, b, state, t, eps))


@cython.wraparound(False)
@cython.boundscheck(False)
cdef calc_gamma(double[:, :] theta, State *state, int i, int k):
    """
    this function calculates the derivative of the scaling factor gamma wrt. theta_ik
    :param theta: matrix containing the theta entries
    :param state: state representing current tumor sample
    :param i: row index of theta entry to take derivative wrt.
    :param k: column index of theta entry to take derivative wrt.
    """
    cdef int mutation_num = get_mutation_num(state)
    cdef int nx = 1 << mutation_num
    cdef int one = 1
    cdef double *deriv_q_diag = <double *> malloc(nx * sizeof(double))
    restricted_derivative_ik_diag(theta, i, state, mutation_num, k, deriv_q_diag)
    cdef double *q_diag = <double*> malloc(nx*sizeof(double))
    restricted_q_diag(theta, state, q_diag)
    cdef double num = ddot(&nx, deriv_q_diag, &one, q_diag, &one)
    cdef double denom = dnrm2(&nx, q_diag, &one)
    free(deriv_q_diag)
    free(q_diag)
    return denom, num / denom


def calc_gamma_wrapper(double[:, :] theta, StateContainer state, int i, int k):
    """
    this function calculates the derivative of the scaling factor gamma wrt. theta_ik. It is a Python wrapper for the
    internal Cython function to compute gamma and its derivative.

    :param theta: matrix containing the theta entries
    :param state: a StateContainer that contains one state for which gamma will be computed
    :param i: row index of theta entry to take derivative wrt.
    :param k: column index of theta entry to take derivative wrt.
    :return: gamma and its derivative wrt. theta_ik
    """
    return calc_gamma(theta, state.states, i, k)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void dua(double[:, :] theta, double[:] b, State *state, double t, int i, int k, double eps, double[:] pt, double[:] dp) except *:
    """
    Computes the frechet derivative of expm(tQ)b using the DUA Algorithm
    :param theta: matrix containing the theta entries
    :param b: distribution to be multiplied from the right
    :param state: state representing the current tumor sample
    :param t: Age of tumor, amount of time to be projected in the future
    :param i: row index of theta to take the derivative wrt.
    :param k: column index of theta to take the derivative wrt.
    :param eps: accuracy
    :param pt: container to store the resulting pt
    :param dp: container to store the derivative wrt. theta_ik
    """
    cdef int mutation_num = get_mutation_num(state)
    cdef int nx = 1 << mutation_num
    cdef int j

    cdef double w = 1.0
    cdef int n = 0 # Iteration number
    cdef int one = 1
    cdef double done = 1.0
    cdef double zero = 0
    cdef double gfac = 1.0
    dscal(&nx, &zero, &pt[0], &one)
    dscal(&nx, &zero, &dp[0], &one)
    cdef double[:] q = b.copy()
    cdef double[:] dq = np.zeros(nx, dtype=np.double)
    cdef double * temp = <double *> malloc(nx * sizeof(double))
    cdef double * temp2 = <double *> malloc(nx * sizeof(double))

    cdef double gamma, dgamma
    gamma, dgamma = calc_gamma(theta, state, i, k)
    cdef double dgam_inv = -1.0/gamma**2*dgamma
    cdef double gam_inv = 1/gamma
    cdef double ewg = exp(-1.0*gamma*t)
    cdef double mass_defect = 0.0
    while eps < (1 - mass_defect):
        mass_defect += ewg
        # pt = pt + exp(-gam*t)q
        daxpy(&nx, &ewg, &q[0], &one, &pt[0], &one)
        # dpt = dpt + exp(-gamma*t)w dq
        daxpy(&nx, &ewg, &dq[0], &one, &dp[0], &one)
        # dpt = dpt + exp(-gamma*t)w dgamma(n/gamma-t)q
        gfac = ewg*dgamma*(n/gamma-t)
        daxpy(&nx, &gfac, &q[0], &one, &dp[0], &one)

        n += 1
        # dq = -1/gamma^2*dg*Q q + 1/gamma*dQ q
        restricted_q_vec(theta, q, state, temp, True, False)
        dscal(&nx, &dgam_inv, temp, &one)
        restricted_derivative_ik(theta, i, q, state, mutation_num, k, temp2)
        daxpy(&nx, &gam_inv, temp2, &one, temp, &one) # temp2 isn't needed anymore and its allocated memory can be reused

        # dq = dq + [1/gamma*Q+I]dq
        restricted_q_vec(theta, dq, state, temp2, True, False)
        daxpy(&nx, &gam_inv, temp2, &one, &dq[0], &one)
        for j in range(nx):
            dq[j] += temp[j] # +temp2[j]

        # q = [1/gamma*Q + I]q
        restricted_q_vec(theta, q, state, temp, True, False)
        daxpy(&nx, &gam_inv, temp, &one, &q[0], &one)

        ewg *= gamma*t/n

    free(temp)
    free(temp2)


cdef bint compute_diff_state(State *former_state, State *current_state, State *diff_state):
    """
    Find all mutations present in the current state that are not in the former state
    
    :param former_state: state of the previous sample
    :param current_state: state of the current sample
    :param diff_state: state that contains only the mutations present in current sample that are not in former sample
    
    :returns: False, if former_state contains a mutations current_state does not contain -> error
    """

    cdef int i
    cdef int former_state_copy, current_state_copy

    for i in range(STATE_SIZE):
        former_state_copy = former_state[0].parts[i]
        current_state_copy = current_state[0].parts[i]
        # check if there is a mutation in the former state that is not in the current state
        if (current_state_copy - former_state_copy) != (current_state_copy ^ former_state_copy):
            return False
        diff_state[0].parts[i] = (current_state_copy ^ former_state_copy)

    return True


cdef compute_modified_theta(double[:, :] theta, double[:, :] modified_theta, State *former_state):
    """
    Modifies theta according to the mutations present in the former sample.
    It sets the base rates of genes that are mutated in the former sample to 0 (this means -inf in log space) and 
    adds the multiplicative effects of those mutations to the base rates of genes that are not mutated in the previous
    sample.
    
    :param theta: actual theta matrix
    :param modified_theta: array that will contain the modified theta at the end
    :param former_state: State object representing the previous sample which is used to compute the modified theta
    """
    cdef int n = theta.shape[0]
    cdef int n_square = n*n
    cdef int one = 1
    cdef int i, j
    cdef int state_copy_i = former_state[0].parts[0]
    cdef int state_copy_j

    # initialize the modified theta with the actual theta matrix
    dcopy(&n_square, &theta[0, 0], &one, &modified_theta[0, 0], &one)

    # look for genes that are mutated in the previous sample and modify all base rates accordingly
    for i in range(n):
        if state_copy_i & 1:
            state_copy_j = former_state[0].parts[0]
            # we want exp(theta_ii) to be 0 if i was mutated in former state
            modified_theta[i, i] = -INFINITY
            # all base rates are amplified by i as i is always mutated
            for j in range(n):
                if not state_copy_j & 1:
                    modified_theta[j, j] += theta[j, i]

                if (j + 1) & 31:
                    state_copy_j >>= 1
                else:
                    state_copy_j = former_state[0].parts[(j + 1) >> 5]

        if (i + 1) & 31:
            state_copy_i >>= 1
        else:
            state_copy_i = former_state[0].parts[(i + 1) >> 5]


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef cython_gradient_and_score(double[:, :] theta, StateAgeContainer mutation_data, double eps):
    """
    This function computes the log-likelihood score and its gradient for a given theta and data, where we know the ages
    of the individual samples (see Rupp et al. (2021) eq. (13)-(15))

    :param theta: theta matrix representing the MHN
    :param mutation_data: data from which we learn the MHN
    :param eps: accuracy
    :returns: the gradient and the score as a tuple
    """
    cdef int n = theta.shape[0]
    cdef int current_nx
    cdef int data_size = mutation_data.data_size
    cdef int k, i, j
    cdef double t
    cdef double score = 0
    cdef double diagonal_partial_grad
    cdef np.ndarray[np.double_t, ndim=2] gradient = np.zeros((n, n), dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] modified_theta = np.empty((n, n))
    cdef np.ndarray[np.double_t] pt
    cdef np.ndarray[np.double_t] dp
    cdef np.ndarray[np.double_t] b
    cdef State diff_state
    cdef int state_copy, state_copy_former

    for k in range(1, data_size):
        # sample k must always be older than sample k-1
        t = mutation_data.state_ages[k] - mutation_data.state_ages[k-1]
        assert t >= 0

        # get the mutations that are new in the current sample and make sure that no mutation has disappeared
        if not compute_diff_state(&mutation_data.states[k-1], &mutation_data.states[k], &diff_state):
            warnings.warn(f"The sample at position {k} contains less mutations than the previous sample")
            continue

        # as we restrict the state-space to contain only states that are "between" the previous and the current state
        # this means that we get the score by multiplying exp(Qt) with (1, 0, ..., 0) and taking the last value
        # (index = current_nx-1) of the resulting vector
        current_nx = 1 << get_mutation_num(&diff_state)
        compute_modified_theta(theta, modified_theta, &mutation_data.states[k-1])
        # For the unlikely case that we have two samples with all genes mutated, the dua algorithm would never be
        # executed. In that case we need pt to be 1 for the score computation at the end, so we initialize pt with ones.
        # If the dua function is called even once, then all pt entries are set to zero there anyway, so this init is fine.
        pt = np.ones(current_nx)
        dp = np.empty(current_nx)
        b = np.zeros(current_nx)
        b[0] = 1
        state_copy_former = mutation_data.states[k-1].parts[0]
        for i in range(n):
            # if the gene was mutated in the previous sample, row i of the gradient matrix is zero everywhere
            if not state_copy_former & 1:
                state_copy = diff_state.parts[0]
                # the gradient of theta_ij is non-zero if gene j is mutated in the current sample
                for j in range(n):
                    if state_copy & 1 or i == j:
                        dua(modified_theta, b, &diff_state, t, i, j, eps, pt, dp)
                        gradient[i, j] += dp[current_nx-1] / pt[current_nx-1]
                        if i == j:
                            diagonal_partial_grad = dp[current_nx-1] / pt[current_nx-1]

                    if (j + 1) & 31:
                        state_copy >>= 1
                    else:
                        state_copy = diff_state.parts[(j + 1) >> 5]

                # if gene j was mutated in the previous sample then the gradient of theta_ij is the same as theta_ii
                state_copy = mutation_data.states[k-1].parts[0]
                for j in range(n):
                    if state_copy & 1:
                        gradient[i, j] += diagonal_partial_grad

                    if (j + 1) & 31:
                        state_copy >>= 1
                    else:
                        state_copy = mutation_data.states[k-1].parts[(j + 1) >> 5]

            if (i + 1) & 31:
                state_copy_former >>= 1
            else:
                state_copy_former = mutation_data.states[k-1].parts[(i + 1) >> 5]

        score += log(pt[current_nx-1])

    return gradient, score