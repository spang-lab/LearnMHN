#ifndef CUDA_FULL_STATE_SPACE_H_
#define CUDA_FULL_STATE_SPACE_H_

// on Windows we need to add a prefix in front of the function we want to use in other code
// on Linux this is not needed, so we define DLL_PREFIX depending on which os this code is compiled on
#ifdef _WIN32
#define DLL_PREFIX __declspec(dllexport)
#else
#define DLL_PREFIX 
#endif


/**
 * this function computes the gradient and score for the current MHN for a given observed frequency of tumors in data using CUDA
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[in] pD observed frequency of tumors in data
 * @param[out] grad_out array of size n*n in which the gradient will be stored
 * @param[out] score_out the marginal log-likelihood score is stored at this position
 *
 * @return CUDA error code converted to integer for better interoperability with Cython
*/
extern "C" int DLL_PREFIX cuda_full_state_space_gradient_score(double *ptheta, int n, double *pD, double *grad_out, double *score_out);


/**
 * Internal function to compute the solution for [I-Q] x = b using forward and backward substitution.
 * All arrays given to this function must be allocated using cudaMalloc()!
 * 
 * @param[in] theta theta matrix representing the MHN with size n x n
 * @param[in] n number of rows and columns of the theta matrix
 * @param[in] dg diagonal of [I-Q], you could also use a different diagonal to compute the inverse for a matrix that only differs in the diagonal from [I-Q]
 * @param[in, out] xout this vector of size 2^n must contain b at the beginning at will contain x at the end
 * @param[in] transp if set to true, computes the solution for [I-Q]^T x = b
*/
extern "C"  void DLL_PREFIX _compute_inverse(const double * __restrict__ theta, const int n, const double * __restrict__ dg, double * __restrict__ xout, bool transp);


/**
 * computes the solution for [I-Q] x = b using forward and backward substitution
 * 
 * @param[in] theta theta matrix representing the MHN with size n x n
 * @param[in] n number of rows and column of the theta matrix
 * @param[in] b vector of size 2^n which should be multiplied with [I-Q]^(-1)
 * @param[out] xout array of size 2^n which will contain the result of the matrix-vector multiplication at the end
*/
extern "C" void DLL_PREFIX gpu_compute_inverse(double *theta, int n, double *b, double *xout);


#endif