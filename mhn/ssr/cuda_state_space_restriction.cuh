#ifndef CUDA_STATE_SPACE_RESTRICTION_H
#define CUDA_STATE_SPACE_RESTRICTION_H


// on Windows we need to add a prefix in front of the function we want to use in other code
// on Linux this is not needed, so we define DLL_PREFIX depending on which os this code is compiled on
#ifdef _WIN32
#define DLL_PREFIX __declspec(dllexport)
#else
#define DLL_PREFIX 
#endif


// this struct is used to store states representing up to 32 * STATE_SIZE genes
// STATE_SIZE must be defined during compilation
typedef struct {
     uint32_t parts[STATE_SIZE];
} State;


// computes the number of mutations present in a given state
/**
 * Computes the number of mutations present in a given state
 *
 * @param[in] state A pointer to a State of which we want to count the number of mutations it contains
*/
int get_mutation_num(const State *state);


/**
 * we determine the number of blocks and threads used in the CUDA kernels for the current data point with this function
 *
 * @param[out] block_num number of blocks that should be used for the CUDA kernels
 * @param[out] thread_num number of threads that should be used for the CUDA kernels
 * @param[in] mutation_num number of mutations present in the current state
*/
inline void determine_block_thread_num(int &block_num, int &thread_num, const int mutation_num);


/**
 * this function is the cuda implementation of the kronvec function for state space restriction
 *
 * IMPORTANT: the result is added to the entries of pout! This makes the q_vec function more efficient. 
 * If you need the result without adding, initialize pout with zeros.
 *
 * @param[in] ptheta array containing the values of theta
 * @param[in] i vector is multiplied with the ith kronecker product (ith summand in eq. 9 of the original paper) 
 * @param[in] px vector that is multiplied with the kronecker product
 * @param[in] state current state used to compute the gradient
 * @param[in] diag if false, the diagonal of the kronecker product is set to zero
 * @param[in] transp if true, the kronecker product is transposed
 * @param[in] n total number of genes considered by the MHN, also column and row size of theta
 * @param[in] mutation_num number of mutations present in state
 * @param[in] count_before_i number of genes mutated that have a lower index than i
 * @param[out] pout vector which will contain the result of this multiplication
*/
__global__ void cuda_restricted_kronvec(const double* __restrict__ ptheta, const int i, const double* __restrict__ px, const State state, const bool diag, const bool transp, const int n, const int mutation_num, int count_before_i, double* __restrict__ pout);


/**
 * computes y = Q(ptheta) * x, result is saved in yout
 *
 * important: ptheta, x and yout must be allocated using cudaMalloc()!
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] x vector that should be multiplied with Q(ptheta)
 * @param[in] state state representing current tumor sample
 * @param[out] yout array in which the result is stored
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[in] mutation_num number of mutations present in the current state / tumor sample
 * @param[in] diag if false, the diag of Q is set to zero during multiplication
 * @param[in] transp if true, multiplication is done with the transposed Q
*/
void cuda_q_vec(const double *ptheta, const double *x, const State *state, double *yout, const int n, const int mutation_num, const bool diag, const bool transp);


/**
 * computes the ith subdiagonal of Q and subtracts(!) it from dg
 * we subtract it, because in jacobi() we need 1 - dg, so dg is initialized with 1 and we subtract the subdiags
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] state state representing current tumor sample
 * @param[in] i this function computes the ith subdiagonal
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[in] mutation_num number of mutations present in the current state / tumor sample
 * @param[in, out] dg the subdiagonal is subtracted from the values in this array
*/
__global__ void cuda_subdiag(const double *ptheta, const State state, const int i, const int n, const int mutation_num, double *dg);


/**
 * subtracts the diag of q from the given dg array, result can be found in dg
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] state state representing current tumor sample
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[in] mutation_num number of mutations present in the current state / tumor sample
 * @param[in, out] dg the subdiagonals are subtracted from the values in this array
 * @param[in] block_num number of blocks used for the CUDA kernels
 * @param[in] thread_num  number of threads used for the CUDA kernels
*/
void cuda_subtract_q_diag(const double *ptheta, const State *state, const int n, const int mutation_num, double *dg, int block_num, int thread_num);


__global__ void fill_array(double *arr, double x, const int size);

__global__ void add_arrays(const double *arr1, double *arr_inout, const int size);

__global__ void divide_arrays_elementwise(const double *arr1, const double *arr2, double *out, const int size);

__global__ void multiply_arrays_elementwise(const double *arr1, double *arr_inout, const int size);

/**
 * this function computes the diagonal of [I-Q] for the jacobi function
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] state state representing current tumor sample
 * @param[in] mutation_num number of mutations present in the current state / tumor sample
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[out] dg this array will contain the diagonal of [I-Q] after calling this function, has size must have size 2^mutation_num
*/
void compute_jacobi_diagonal(const double* ptheta, const State* state, const int mutation_num, const int n, double* dg);


/**
 * this functions multiplies [I-Q]^(-1) with b
 * all arrays given to this function must be allocated using cudaMalloc()
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] b array that is multiplied with [I-Q]^(-1)
 * @param[in] state state representing current tumor sample
 * @param[in] mutation_num number of mutations present in the current state / tumor sample
 * @param[in] transp if true, b is multiplied with the transposed [I-Q]^(-1)
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[out] xout the results of this function are stored in this array
 * @param[in, out] tmp this array is used to store temporary data, has to have size 2^mutation_num
 * @param[in] dg this array contains the diagonal of [I-Q]
*/
void cuda_jacobi(const double *ptheta, const double *b, const State *state, const int mutation_num, const bool transp, const int n, double *xout, double *tmp, double *dg);


/**
 * this functions shuffles the entries of old_vec into the entries of to_shuffle_vec
 *
 * @param[in] old_vec array that should be shuffled
 * @param[out] to_shuffle_vec array in which the shuffled vector is stored
 * @param[in] nx size of both vectors
*/
__global__ void shuffle(const double* __restrict__ old_vec, double* __restrict__ to_shuffle_vec, const int nx);


/**
 * inspired by https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * computes the sum of all entries in a given array
*/
__global__ void sum_over_array(const double *arr, double *result, int size);


/**
 * function for debugging purposes
*/
__global__ void print_vec(double *vec, int size);


/**
 * compute the gradient for one tumor sample
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] state state representing current tumor sample
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[out] grad array that will contain the gradient at the end, size n*n
 * @param[in] p0_pD memory buffer needed for this function, size 2^mutation_num
 * @param[in] pth memory buffer needed for this function, size 2^mutation_num
 * @param[in] q memory buffer needed for this function, size 2^mutation_num
 * @param[in] tmp1 memory buffer needed for this function, size 2^mutation_num
 * @param[in] tmp2 memory buffer needed for this function, size 2^mutation_num
*/
void cuda_restricted_gradient(const double *ptheta, const State *state, const int n, double *grad, double *p0_pD, double *pth, double *q, double *tmp1, double *tmp2);

__global__ void array_exp(double *arr, int size);

__global__ void add_to_score(double *score, double *pth_end);


extern "C"
{
    /**
     * this function computes the gradient and score for the current MHN for a given data set using CUDA
     *
     * @param[in] ptheta array containing the theta entries
     * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
     * @param[in] mutation_data array of States, where each state represents a tumor sample
     * @param[in] data_size number of tumor samples in mutation_data
     * @param[out] grad_out array of size n*n in which the gradient will be stored
     * @param[out] score_out the marginal log-likelihood score is stored at this position
     *
     * @return CUDA error code converted to integer for better interoperability with Cython
    */
    int DLL_PREFIX cuda_gradient_and_score_implementation(double *ptheta, int n, State *mutation_data, int data_size, double *grad_out, double *score_out);

    /**
     * This function is used by state_space_restriction.pyx to get the error name and description if an error occurred
     *
     * @param[in] error is the cudaError_t returned by the CUDA function casted to int to be usable in Cython
     * @param[out] error_name the name of the error will be stored in this variable
     * @param[out] error_description the description of the error will be stored in this variable
    */
    void DLL_PREFIX get_error_name_and_description(int error, const char **error_name, const char **error_description);


    /**
     * This function can be used to check if CUDA works as intended. For that it allocates and frees memory on the GPU.
     * If the allocation fails, something is probably wrong with the CUDA drivers and you should check your CUDA installation.
     *
     * @return 1, if everything works as it should, else 0
    */
    int DLL_PREFIX cuda_functional();
}
#endif