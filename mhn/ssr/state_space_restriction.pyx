# distutils: language = c++

# by Stefan Vocht
#
# implement StateSpaceRestriction using Cython
#

cimport cython

from scipy.linalg.cython_blas cimport dcopy, dscal, daxpy, ddot
from libc.stdlib cimport malloc, free
from cython.parallel import parallel, prange
from libc.math cimport exp, log

from mhn.ssr.state_storage cimport State, StateStorage

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from *:
    """
    /*
    Counts number of 1s in binary representation of number x, where x is a 32-bit integer
    Source: https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
    */
    int count_ones32(uint32_t i){
        i = i - ((i >> 1) & 0x55555555);                                    // add pairs of bits
        i = (i & 0x33333333) + ((i >> 2) & 0x33333333);                     // quads
        i = (i + (i >> 4)) & 0x0F0F0F0F;                                    // groups of 8
        return (i * 0x01010101) >> 24;                                      // horizontal sum of bytes
    }

    /*
    Counts number of 1s in binary representation of number x, where x is a 64-bit integer
    Source: https://en.wikipedia.org/wiki/Hamming_weight 
    */
    int count_ones(long long x) {
        x -= (x >> 1) & 0x5555555555555555LL;             					//put count of each 2 bits into those 2 bits
        x = (x & 0x3333333333333333LL) + ((x >> 2) & 0x3333333333333333LL); //put count of each 4 bits into those 4 bits 
        x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fLL;        					//put count of each 8 bits into those 8 bits 
        return (x * 0x0101010101010101LL) >> 56;  							//returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ... 
    }
    """
    inline int count_ones32(unsigned int u) nogil
    inline int count_ones(long long x) nogil


def count_ones64(long long x):
    """
    Wrapper so that count_ones can be called from a Python script
    """
    return count_ones(x)


cdef int get_mutation_num(State *state):
    """
    Get the number of mutations in a given state
    """
    cdef int mutation_num = 0
    cdef int i

    for i in range(STATE_SIZE):
        mutation_num += count_ones32(state[0].parts[i])

    return mutation_num


# load the function cuda_gradient_and_score if the CUDA compiler is available
IF NVCC_AVAILABLE:
    cdef extern from *:
        """
        #ifdef _WIN32
        #define DLL_PREFIX __declspec(dllexport)
        #else
        #define DLL_PREFIX 
        #endif

        double DLL_PREFIX cuda_gradient_and_score(double *ptheta, int n, State *mutation_data, int data_size, double *grad_out);
        int DLL_PREFIX cuda_functional();
        """

        double cuda_gradient_and_score(double *ptheta, int n, State *mutation_data, int data_size, double *grad_out)
        int cuda_functional()


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void restricted_kronvec(double[:, :] theta_mat, int i, double[:] x_vec, State *state, int mutation_num, double *pout, bint diag = False, bint transp = False) nogil:
    """
    This function multiplies the kronecker product described in the original MHN paper in eq.9 with a vector

    :param theta_mat: matrix containing the theta entries
    :param i: vector is multiplied with the ith kronecker product (ith summand in eq. 9 of the original paper) 
    :param x_vec: vector that is multiplied with the kronecker product
    :param state: current state used to compute the gradient
    :param mutation_num: number of mutations present in state
    :param pout: vector which will contain the result of this multiplication
    :param diag: if False, the diagonal of the kronecker product is set to zero
    :param transp: if True, the kronecker product is transposed
    """

    # inizialize some constants used in this function
    cdef double[:] theta_i = theta_mat[i, :]
    cdef int n = theta_i.shape[0]
    cdef int nx = 1 << mutation_num
    cdef int nxhalf = nx / 2
    cdef double mOne = -1
    cdef double zero = 0

    cdef int incx = 1
    cdef int incx2 = 2
    cdef int j

    # if we have no diagonal and the ith gene is not mutated, the result is always a zero vector
    if not diag and not (state[0].parts[i >> 5] >> (i & 31)) & 1:
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


cdef void restricted_q_vec(double[:, :] theta, double[:] x, State *state, double *yout, bint diag= False, bint transp = False):
    """
    computes y = Q(ptheta) * x, result is saved in yout

    :param theta: matrix containing the theta entries
    :param x: vector that should be multiplied with Q(ptheta)
    :param state: state representing current tumor sample
    :param yout: array in which the result is stored
    :param diag: if False, the diag of Q is set to zero during multiplication
    :param transp: if True, multiplication is done with the transposed Q
    """

    cdef int n = theta.shape[0]
    cdef int nx = x.shape[0]
    cdef int i
    cdef double one_d = 1
    cdef int one_i = 1
    cdef double zero = 0

    # get the number of mutations present in the given state
    cdef int mutation_num = get_mutation_num(state)

    cdef double *result_vec = <double *> malloc(sizeof(double) * nx)

    # initialize yout with zero
    dscal(&nx, &zero, yout, &one_i)

    for i in range(n):
        restricted_kronvec(theta, i, x, state, mutation_num, result_vec, diag, transp)
        # add result of restricted_kronvec to yout
        daxpy(&nx, &one_d, result_vec, &one_i, yout, &one_i)

    free(result_vec)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void restricted_q_diag(double[:, :] theta, State *state, double *dg):
    """
    Compute the diagonal of the transition rate matrix Q

    :param theta: matrix containing the theta entries
    :param state: state representing current tumor sample
    :param dg: array in which the diagonal is stored at the end
    """
    cdef int mutation_num = get_mutation_num(state)
    cdef int nx = 1 << mutation_num
    cdef int n = theta.shape[0]
    cdef int i, j
    cdef int state_copy

    cdef double *s = <double *> malloc(nx * sizeof(double))
    cdef int current_length
    cdef double exp_theta
    cdef double d_one = 1
    cdef double zero = 0
    cdef int i_one = 1

    # initialize the diagonal with zero
    dscal(&nx, &zero, dg, &i_one)

    # compute the ith subdiagonal of Q and add it to dg
    for i in range(n):
        state_copy = state[0].parts[0]
        current_length = 1
        s[0] = 1
        # compute the ith subdiagonal of Q 
        for j in range(n):
            if state_copy & 1:
                exp_theta = exp(theta[i, j])
                if i == j:
                    exp_theta *= -1
                    dscal(&current_length, &exp_theta, s, &i_one)
                    dscal(&current_length, &zero, s + current_length, &i_one)
                else:
                    dcopy(&current_length, s, &i_one, s + current_length, &i_one)
                    dscal(&current_length, &exp_theta, s + current_length, &i_one)

                current_length *= 2

            elif i == j:
                exp_theta = - exp(theta[i, j])
                dscal(&current_length, &exp_theta, s, &i_one)

            if (j + 1) & 31:
                state_copy >>= 1
            else:
                state_copy = state[0].parts[(j+1) >> 5]

        # add the subdiagonal to dg
        daxpy(&nx, &d_one, s, &i_one, dg, &i_one)

    free(s)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef double [:] restricted_jacobi(double[:, :] theta, double[:] b, State *state, bint transp = False):
    """
    this functions multiplies [I-Q]^(-1) with b

    :param theta: matrix containing the theta entries
    :param b: array that is multiplied with [I-Q]^(-1)
    :param state: state representing current tumor sample
    :param transp: if True, b is multiplied with the tranposed [I-Q]^(-1)
    """
    cdef int mutation_num = get_mutation_num(state)
    cdef int nx = 1 << mutation_num
    cdef int z, j
    cdef int i_one = 1
    cdef int zero = 0
    cdef double d_one = 1
    cdef double mOne = -1

    cdef double[:] x = np.full(nx, 1 / (1.0 * nx), dtype=np.float)
    cdef double *q_vec_result = <double *> malloc(nx * sizeof(double))

    # compute the diagonal of [I-Q], store it in dg
    cdef double *dg = <double *> malloc(nx * sizeof(double))
    restricted_q_diag(theta, state, dg)
    daxpy(&nx, &d_one, &mOne, &zero, dg, &i_one)
    dscal(&nx, &mOne, dg, &i_one)

    for z in range(mutation_num+1):
        restricted_q_vec(theta, x, state, q_vec_result, diag=False, transp=transp)
        # add b to the result of q_vec
        daxpy(&nx, &d_one, &b[0], &i_one, q_vec_result, &i_one)
        # divide every entry by its corresponding diagonal entry
        for j in range(nx):
            x[j] = q_vec_result[j] / dg[j]

    free(dg)
    free(q_vec_result)
    return x


@cython.wraparound(False)
@cython.boundscheck(False)
cdef double restricted_gradient_and_score(double[:, :] theta, State *state, double[:, :] g):
    """
    Computes a part of the gradient and score corresponding to a given state

    :param theta: matrix containing the theta entries
    :param state: state representing current tumor sample
    :param g: the resulting gradient is stored in this matrix
    :return: part of the total score
    """
    cdef int n = theta.shape[0]
    cdef int mutation_num = get_mutation_num(state)
    cdef int nx = 1 << mutation_num
    cdef int nxhalf = nx / 2
    p0 = np.zeros(nx, dtype=np.float) 
    p0[0] = 1

    # compute parts of the probability distribution yielded by the current MHN
    cdef double[:] pth = restricted_jacobi(theta, p0, state)

    pD = np.zeros(nx) 
    pD[nx-1] = 1 / pth[nx-1]

    cdef double[:] q = restricted_jacobi(theta, pD, state, transp=True)

    cdef int i, j

    for i in range(n):
        for j in range(n):
            g[i, j] = 0

    cdef int state_copy
    cdef double *r_vec = <double *> malloc(nx * sizeof(double))

    cdef double *shuffled_vec
    cdef double *old_vec
    cdef double *swap_vec
    cdef int incx = 1
    cdef int incx2 = 2
    cdef int incx0 = 0
    cdef double one = 1.

    cdef double *ptmp = <double *> malloc(nx * sizeof(double))

    # compute the gradient efficiently using the shuffle trick
    for i in range(n):
        restricted_kronvec(theta, i, pth, state, mutation_num, ptmp, diag=True)
        for j in range(nx):
            r_vec[j] = q[j] * ptmp[j] 
        old_vec = &r_vec[0]
        shuffled_vec = ptmp  # reuse ptmp for the shuffle as it is already allocated memory
        state_copy = state[0].parts[0]
        for j in range(n):
            if state_copy & 1:
                dcopy(&nxhalf, old_vec, &incx2, shuffled_vec, &incx)
                dcopy(&nxhalf, old_vec+1, &incx2, shuffled_vec+nxhalf, &incx)
                g[i, j] = ddot(&nxhalf, shuffled_vec+nxhalf, &incx, &one, &incx0)
                if i == j:
                    g[i, j] += ddot(&nxhalf, shuffled_vec, &incx, &one, &incx0)

                swap_vec = old_vec
                old_vec = shuffled_vec
                shuffled_vec = swap_vec

            elif i == j:
                g[i, j] = ddot(&nx, old_vec, &incx, &one, &incx0)

            if (j + 1) & 31:
                state_copy >>= 1
            else:
                state_copy = state[0].parts[(j+1) >> 5]

    free(ptmp)
    free(r_vec)

    return log(pth[nx - 1])


cpdef cython_gradient_and_score(double[:, :] theta, StateStorage mutation_data):
    """
    Computes the total gradient and score for a given MHN and given mutation data

    :param theta: matrix containing the theta entries of the current MHN
    :param mutation_data: StateStorage containing the mutation data the MHN should be trained on
    :return: tuple containing the gradient and the score
    """
    cdef int n = theta.shape[0]
    cdef int data_size = mutation_data.data_size
    cdef int i, j
    final_gradient = np.zeros((n, n))
    cdef double *local_grad_sum
    cdef double [:, :] local_gradient_container = np.empty((n, n), dtype=np.float)
    cdef double zero = 0
    cdef int incx = 1
    cdef double one = 1
    cdef int n_square = n*n

    cdef double score = 0

    for i in range(data_size):
        score += restricted_gradient_and_score(theta, &mutation_data.states[i], local_gradient_container)
        final_gradient += local_gradient_container

    return (final_gradient / data_size, score / data_size)


# this function is only defined if the CUDA-compiler (nvcc) is available on your device
IF NVCC_AVAILABLE:
    cpdef gradient_and_score_with_cuda(double[:, :] theta, StateStorage mutation_data):
        """
        This function is a wrapper for the cuda implementation of the state space restriction

        :param theta: matrix containing the theta entries of the current MHN
        :param mutation_data: StateStorage containing the mutation data the MHN should be trained on
        :return: tuple containing the gradient and the score
        """

        cdef int n = theta.shape[0]
        cdef int data_size = mutation_data.data_size

        cdef double[:] grad_out = np.empty(n * n)

        cdef double score = cuda_gradient_and_score(&theta[0, 0], n, &mutation_data.states[0], data_size, &grad_out[0])

        return (np.asarray(grad_out).reshape((n, n)) / data_size, score / data_size)


cpdef gradient_and_score(double[:, :] theta, StateStorage mutation_data):
    """
    This function computes the gradient using Cython AND CUDA (only if CUDA is installed)
    It will compute the gradients for data points with few mutations using the Cython implementation
    and compute the gradients for data points with many mutations using CUDA.
    If CUDA is not installed on your device, this function will only use the Cython implementation.
    """
    IF NVCC_AVAILABLE:
        cdef int data_size = mutation_data.data_size
        cdef int n = theta.shape[0]
        cdef State *sorted_data = <State*> malloc(data_size * sizeof(State));

        cdef double *grad_out

        cdef int index_left, index_right
        cdef int i, j
        cdef State *state
        cdef int mutation_num

        cdef double score = 0

        # number of mutations for which it is still faster to do it in cython
        # than using the cuda implementation
        # may vary depending on your device
        DEF critical_size = 13

        index_left = 0
        index_right = data_size - 1

        # sort the data into samples that have more, and samples that have less 
        # than *critical_size* mutations
        for i in range(data_size):
            mutation_num = get_mutation_num(&mutation_data.states[i])
            if(mutation_num > critical_size):
                sorted_data[index_right] = mutation_data.states[i]
                index_right -= 1
            else:
                sorted_data[index_left] = mutation_data.states[i]
                index_left += 1

        cdef cnp.ndarray[cnp.float_t, ndim=2]  final_gradient = np.zeros((n, n), dtype=np.float)
        cdef cnp.ndarray[cnp.float_t, ndim=2]  tmp_gradient = np.zeros((n, n), dtype=np.float)

        for i in range(index_left):
            state = &sorted_data[i]
            score += restricted_gradient_and_score(theta, state, tmp_gradient)
            final_gradient += tmp_gradient

        # check if there is any data point with more than *critical_size* mutations
        # and only call the CUDA function if this is the case
        if index_right != data_size - 1:
            grad_out = <double *> malloc(n*n * sizeof(double))
            score += cuda_gradient_and_score(&theta[0, 0], n, sorted_data + index_right + 1, data_size - index_left, grad_out)

            for i in range(n):
                for j in range(n):
                    final_gradient[i, j] += grad_out[i*n + j]

            free(grad_out)

        free(sorted_data)
        return (final_gradient / data_size, score / data_size)

    ELSE:
        return cython_gradient_and_score(theta, mutation_data)


CUDA_AVAILABLE = "CUDA is available"
CUDA_NOT_AVAILABLE = "The CUDA compiler nvcc could not be found"
CUDA_NOT_FUNCTIONAL = "CUDA compiler nvcc available but CUDA functions not working. Check CUDA installation"
def cuda_available():
    """
    Call this function if you want to know if the mhn package is able to use CUDA functions on your device.
    """
    IF NVCC_AVAILABLE:
        if cuda_functional():
            return CUDA_AVAILABLE
        else:
            return CUDA_NOT_FUNCTIONAL
    ELSE:
        return CUDA_NOT_AVAILABLE

