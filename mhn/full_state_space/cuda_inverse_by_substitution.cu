// by Stefan Vocht
// this file contains CUDA functions to compute the solution for [I-Q] x = b using forward and backward substitutions


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <math.h>


// on Windows we need to add a prefix in front of the function we want to use in other code
// on Linux this is not needed, so we define DLL_PREFIX depending on which os this code is compiled on
#ifdef _WIN32
#define DLL_PREFIX __declspec(dllexport)
#else
#define DLL_PREFIX
#endif


// small function to compute n choose k
int compute_binom_coef(int n, int k){
    if(k>n || k < 0)
        return 0;
    int res = 1;
    if (k > n - k)
        k = n-k;

    for(int i = 0; i < k; ++i){
        res *= (n-i);
        res /= (i+1);
    }
    return res;
}


/**
 * this function represents a bijective mapping between the numbers 0 to (n choose k) and the numbers which are smaller than 2^n and
 * contain exactly k 1s in their binary representation 
 * 
 * @param[in] i number which should be mapped to a bit permutation with n bits and k 1s
 * @param[in] n considers permutations with n bits
 * @param[in] k number of bits set to 1 in the permutations
 * @param[in] binom_coef number of possible permutations for n bits with k bits set to 1
 * 
 * @return permutation corresponding to the number i
*/
__device__ int compute_index(int i, int n, int k, int binom_coef){
    int index = 0;
    int bit_setter = 1;
    int current_n = n;

    // the algorithm can be imagined as a kind of binary tree search:
    // in each iteration j there are two possiblities: either the jth bit is set to 0 or to 1
    // if k bits are set to 1, we are finished
    // let current_n be the number of bits not determined yet (current_n = n - j)
    // let k be the number of bits that still have to be flipped to 1
    // then we can compute the size of each subtree with 
    // (current_n-1 choose k) for the subtree where we do not set the current bit to 1 and
    // (current_n-1 choose k-1) for the subtree where we do set the current bit to 1
    // if the given number i is greater than the size of the subtree where the bit is not set to 1, we set the bit to 1 and
    // subtract the size of that subtree from i for the next iteration

    for(int j = 0; j < n; j++){
        // compute (current_n-1 choose k)
        binom_coef = ((current_n-k) * binom_coef) / current_n;
        if (i >= binom_coef) {
            index |= bit_setter;
            i -= binom_coef;
            // compute (current_n-1 choose k-1)
            if (current_n == k){
                binom_coef = 1;
            } else {
                binom_coef = (k * binom_coef) / (current_n - k);
            }           
            k -= 1;
            if(k == 0)
                break;
        }
        current_n -= 1;
        bit_setter <<= 1;
    }

    return index;
}


/**
 * this kernel is used to solve [I - Q] x = b for all indices whose binary representation contains j bits set to 1
 * 
 * @param[in] theta theta matrix representing the MHN
 * @param[in] n size of theta
 * @param[in] dg diagonal of [I - Q]
 * @param[in, out] xout array containing the b at the beginning and x at the end
 * @param[in] j number of bits set to 1 in all indices for which the equation is solved
 * @param[in] binom_coef value of n choose j
*/
__global__ void compute_inverse_level(const double * __restrict__ theta, const int n, const double * __restrict__ dg, double * __restrict__ xout, int j, int binom_coef){
    const int stride = blockDim.x * gridDim.x;
    const int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double local_theta[];

    for(int i = threadIdx.x; i < n*n; i += blockDim.x){
        local_theta[i] = theta[i];
    }

    __syncthreads();

    // in each iteration we first map the current i to a permutation of bits with j 1s and n-j 0s
    // for the indices represented by those permutations all partial solutions needed to compute their values were computed with forward substitution in a previous call of this kernel
    // we can therefore now go on and compute the solution for those indices
    for(int i = cuda_index; i < binom_coef; i += stride){
        int index = compute_index(i, n, j, binom_coef);
        int bit_setter = 1;
        double xout_element = xout[index];
        for(int k = 0; k < n; k++){
            int modified_index = (index & (~bit_setter));
            if (modified_index != index){
                double theta_product = 1.;
                int ind_copy = index;
                for(int r = 0; r < n; r++){
                    theta_product *= 1 + (ind_copy & 1) * (local_theta[k*n + r] - 1);
                    ind_copy >>= 1;
                }
                // index was chosen in such a way that we can be sure that xout[modified_index] already contains the correct value
                xout_element += theta_product * xout[modified_index];
            }
            bit_setter <<= 1;
        }
        xout[index] = xout_element / dg[index];
    }
}


/**
 * this kernel is used to solve [I - Q]^T x = b for all indices whose binary representation contains j bits set to 1
 * 
 * @param[in] theta theta matrix representing the MHN
 * @param[in] n size of theta
 * @param[in] dg diagonal of [I - Q]
 * @param[in, out] xout array containing the b at the beginning and x at the end
 * @param[in] j number of bits set to 1 in all indices for which the equation is solved
 * @param[in] binom_coef value of n choose j
*/
__global__ void compute_inverse_level_t(const double * __restrict__ theta, const int n, const double * __restrict__ dg, double * __restrict__ xout, int j, int binom_coef){
    const int stride = blockDim.x * gridDim.x;
    const int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double local_theta[];

    for(int i = threadIdx.x; i < n*n; i += blockDim.x){
        local_theta[i] = theta[i];
    }

    __syncthreads();

    // in each iteration we first map the current i to a permutation of bits with j 1s and n-j 0s
    // for the indices represented by those permutations all partial solutions needed to compute their values were computed with forward substitution in a previous call of this kernel
    // we can therefore now go on and compute the solution for those indices
    for(int i = binom_coef - cuda_index - 1; i >= 0; i -= stride){
        int index = compute_index(i, n, j, binom_coef);
        int bit_setter = 1;
        double xout_element = xout[index];
        for(int k = 0; k < n; k++){
            int modified_index = (index | bit_setter);
            if (modified_index != index){
                double theta_product = 1.;
                int ind_copy = modified_index;
                for(int r = 0; r < n; r++){
                    theta_product *= 1 + (ind_copy & 1) * (local_theta[k*n + r] - 1);
                    ind_copy >>= 1;
                }
                // index was chosen in such a way that we can be sure that xout[modified_index] already contains the correct value
                xout_element += theta_product * xout[modified_index];
            }
            bit_setter <<= 1;
        }
        xout[index] = xout_element / dg[index];
    }
}

/**
 * Internal function to compute the solution for [I-Q] x = b using forward and backward substitution
 * All arrays given to this function must be allocated using cudaMalloc()! 
 * 
 * @param[in] theta theta matrix representing the MHN with size n x n
 * @param[in] n number of rows and columns of the theta matrix
 * @param[in] dg diagonal of [I-Q], you could also use a different diagonal to compute the inverse for a matrix that only differs in the diagonal from [I-Q]
 * @param[in, out] xout this vector of size 2^n must contain b at the beginning at will contain x at the end
 * @param[in] transp if set to true, computes the solution for [I-Q]^T x = b
*/
extern "C"  void DLL_PREFIX _compute_inverse(const double * __restrict__ theta, const int n, const double * __restrict__ dg, double * __restrict__ xout, bool transp){

    for(int j = 0; j <= n; j++){
        int binom_coef = compute_binom_coef(n, j);
        int thread_num = ((binom_coef / 32) + 1) * 32;
        if (thread_num > 512)
            thread_num = 512;
        int block_num = 1 + (binom_coef / thread_num);
        if (block_num > 128)
            block_num = 128;
        if(transp){
            compute_inverse_level_t<<<block_num, thread_num, n*n * sizeof(double)>>>(theta, n, dg, xout, n-j, binom_coef);
        } else {
            compute_inverse_level<<<block_num, thread_num, n*n * sizeof(double)>>>(theta, n, dg, xout, j, binom_coef);
        }
    }
}
