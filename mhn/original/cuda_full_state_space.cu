
// by Stefan Vocht
// this file contains CUDA functions to compute
// the gradients for training a MHN on full state-space


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


/**
 * we determine the number of blocks and threads used in the CUDA kernels for the current data point with this function
 *
 * @param[out] block_num number of blocks that should be used for the CUDA kernels
 * @param[out] thread_num number of threads that should be used for the CUDA kernels
 * @param[in] n size of the MHN
*/
inline void determine_block_thread_num(int &block_num, int &thread_num, const int n) {

	// block_num and thread_num have to be powers of two, else cuda_kronvec will not work
	// maximum 256 blocks with 1024 threads
	if (n >= 17) {
		block_num = 256;
		thread_num = 512;
	}
	// define a minimum number of blocks and threads per block
	else if (n < 12) {
		block_num = 32;
		thread_num = 64;
	}
	else {
		block_num = 1 << (n / 2);
		thread_num = 1 << (n / 2 + (n & 1));
	}
}

/**
 * this function is the cuda implementation of the kronvec function for full state-space
 *
 * IMPORTANT: the result is added to the entries of pout! This makes the q_vec function more efficient.
 * If you need the result without adding, initialize pout with zeros.
 *
 * @param[in] ptheta array containing the values of theta
 * @param[in] i vector is multiplied with the ith kronecker product (ith summand in eq. 9 of the original paper)
 * @param[in] px vector that is multiplied with the kronecker product
 * @param[in] diag if false, the diagonal of the kronecker product is set to zero
 * @param[in] transp if true, the kronecker product is transposed
 * @param[in] n total number of genes considered by the MHN, also column and row size of theta
 * @param[out] pout vector which will contain the result of this multiplication
*/
__global__ void cuda_kronvec(const double* __restrict__ ptheta, const int i, const double* __restrict__ px, const bool diag, const bool transp, const int n, double* __restrict__ pout) {
	const int stride = blockDim.x * gridDim.x;
	const int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	// in the following 1 << i is equivalent to 2^i, x >> i is equivalent to x // 2^i, x & ((1<<i)-1) to x % 2^i
	const int nx = 1 << n;

	extern __shared__ double theta_i[];

	// load the ith row of theta into shared memory for more efficient access
	for (int j = threadIdx.x; j < n; j += blockDim.x) {
		theta_i[j] = ptheta[i * n + j];
	}

	__syncthreads();

	// patch_size is important for later for the case i == j in the shuffle algorithm
	// as we do not actually shuffle the data in px in this implementation (only implicitly), we have to keep track of some indices
	// and which entries have to be computed together in the case i == j. Those two indices are (x_index) and (x_index + patch_size)
	// ("patch_size", as over all, the entries that have to be computed together occur in patches of size 2**(count_before_i))
	const int patch_size = 1 << i;
	int x_index = ((cuda_index >> i) << (i + 1)) + (cuda_index & (patch_size - 1));

	// for each iteration of this while loop, we compute the output values for indices (x_index) and (x_index + patch_size)
	// and add the result to pout
	while (x_index + patch_size < nx) {
		// for each entry the theta_ij that have to be multiplied to give us the correct result are given
		// by the bit representation of its index:
		// if the kth bit of the index is set to 1 we have to use theta_ik to compute the output
		// as patch_size is a power of two, (x_index) and (x_index + patch_size) only differ in a single bit,
		// namely the ith one
		double theta_product = 1.;

		int x_index_copy = x_index;
		double theta;

		for (int j = 0; j < n; j++) {
            theta = theta_i[j];
            if (i == j) {
                // if i == j then that theta is always part of theta_product
                theta_product *= theta;
            }
            else {
                // if the current first bit of x_index_copy is set to 1, multiply with theta
                // else multiply with one
                // here the if condition is implicitly in the computation to avoid branching of the threads
                theta_product *= 1. + (x_index_copy & 1) * (theta - 1.);
            }
            // shift the bits by one for the next iteration
            x_index_copy >>= 1;
		}

		// we now have to make computations involving the entries (x_index) and (x_index + patch_size)
		// this is the part for which it was important to choose the correct patch_size and why we needed to compute two entries at once
		// the following computations follow from the part of the shuffle algorithm where we multiply the 2x2 matrix containing theta_ii with px
        if (!transp) {
            double output = px[x_index] * theta_product;
            pout[x_index + patch_size] += output;
            if (diag) {
                pout[x_index] -= output;
            }
        }
        else {
            if (diag) {
                // this case never occurs during gradient computation, its just here for the sake of completeness
                pout[x_index] += (px[x_index + patch_size] - px[x_index]) * theta_product;
            }
            else {
                pout[x_index] += px[x_index + patch_size] * theta_product;
            }
        }


		// if patch_size is bigger than stride, we have to do corrections to the indices
		if (stride < patch_size) {
			// check if the current index is inside an odd patch, if so, jump to the next one
			x_index += stride;
			x_index += ((x_index >> i) & 1) * patch_size;
		}
		else {
			x_index += 2 * stride;
		}
	}
}


/**
 * computes y = Q(ptheta) * x, result is saved in yout
 *
 * important: ptheta, x and yout must be allocated using cudaMalloc()!
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] x vector that should be multiplied with Q(ptheta)
 * @param[out] yout array in which the result is stored
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[in] diag if false, the diag of Q is set to zero during multiplication
 * @param[in] transp if true, multiplication is done with the transposed Q
*/
static void cuda_q_vec(const double* __restrict__ ptheta, const double* __restrict__ x, double* __restrict__ yout, const int n, const bool diag, const bool transp) {

	const int nx = 1 << n;
	cudaMemset(yout, 0, nx * sizeof(double));

	int block_num, thread_num;

	determine_block_thread_num(block_num, thread_num, n);

	for (int i = 0; i < n; i++) {
		cuda_kronvec<<<block_num, thread_num, n * sizeof(double)>>>(ptheta, i, x, diag, transp, n, yout);
	}
}


/**
 * computes the ith subdiagonal of Q and subtracts(!) it from dg
 * we subtract it, because in jacobi() we need 1 - dg, so dg is initialized with 1 and we subtract the subdiags
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] i this function computes the ith subdiagonal
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[in, out] dg the subdiagonal is subtracted from the values in this array
*/
__global__ void cuda_subdiag(const double* __restrict__ ptheta, const int i, const int n, double* __restrict__ dg) {
	int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	// in the following 1 << i is equivalent to 2^i, x >> i is equivalent to x // 2^i, x & ((1<<i)-1) to x % 2^i
	const int nx = 1 << n;

	// store the ith row of theta in shared memory for more efficient access
	extern __shared__ double theta_i[];

	for(int j = threadIdx.x; j < n; j += blockDim.x){
		theta_i[j] = ptheta[i*n + j];
	}
	__syncthreads();


	for (int k = cuda_index; k < nx; k += stride) {

		double dg_entry = 1;

		int position_condition = k;
		for (int j = 0; j < n; j++) {
			double theta = theta_i[j];
			// depending on the index different thetas have to be multiplied to the subdiag entry
            if (i == j) {
                dg_entry *= -(1 - (position_condition & 1)) * theta;
            }
            else {
                dg_entry *= 1 + (position_condition & 1) * (theta - 1);
            }
            position_condition >>= 1;
		}
		//subtract the subdiagonal from the diagonal entries
		dg[k] -= dg_entry;
	}
}


/**
 * subtracts the diag of Q from the given dg array, result can be found in dg
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[in, out] dg the subdiagonals are subtracted from the values in this array
 * @param[in] block_num number of blocks used for the CUDA kernels
 * @param[in] thread_num  number of threads used for the CUDA kernels
*/
static void cuda_subtract_q_diag(const double* __restrict__ ptheta, const int n, double* __restrict__ dg, int block_num, int thread_num) {
	for (int i = 0; i < n; i++) {
		cuda_subdiag<<<block_num, thread_num, n * sizeof(double)>>>(ptheta, i, n, dg);
	}
}


__global__ void fill_array(double *arr, double x, const int size){
	int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = cuda_index; k < size; k += stride) {
		arr[k] = x;
	}
}

__global__ void add_arrays(const double *arr1, double *arr_inout, const int size) {
	int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = cuda_index; k < size; k += stride) {
		arr_inout[k] += arr1[k];
	}
}

__global__ void divide_arrays_elementwise(const double *arr1, const double *arr2, double *out, const int size) {
	int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = cuda_index; k < size; k += stride) {
		out[k] = arr1[k] / arr2[k];
	}
}

__global__ void multiply_arrays_elementwise(const double *arr1, double *arr_inout, const int size) {
	int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = cuda_index; k < size; k += stride) {
		arr_inout[k] *= arr1[k];
	}
}

/**
 * this function computes the diagonal of [I-Q] for the jacobi function
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[out] dg this array will contain the diagonal of [I-Q] after calling this function, has size must have size 2^mutation_num
*/
static void compute_jacobi_diagonal(const double* __restrict__ ptheta, const int n, double* __restrict__ dg) {
	const int nx = 1 << n;

	int block_num, thread_num;
	determine_block_thread_num(block_num, thread_num, n);

	// initialize the diagonal entries
	fill_array <<<block_num, thread_num >>> (dg, 1, nx);
	cuda_subtract_q_diag(ptheta, n, dg, block_num, thread_num);
}

/**
 * this functions multiplies [I-Q]^(-1) with b
 * all arrays given to this function must be allocated using cudaMalloc()
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] b array that is multiplied with [I-Q]^(-1)
 * @param[in] transp if true, b is multiplied with the transposed [I-Q]^(-1)
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[out] xout the results of this function are stored in this array
 * @param[in, out] tmp this array is used to store temporary data, has to have size 2^n
 * @param[in] dg this array contains the diagonal of [I-Q]
*/
static void cuda_jacobi(const double* __restrict__ ptheta, const double* __restrict__ b, const bool transp, const int n, double* __restrict__ xout, double* __restrict__ tmp, double* __restrict__ dg) {

	const int nx = 1 << n;

	int block_num, thread_num;
	determine_block_thread_num(block_num, thread_num, n);

    // initialize the entries of xout with 1/nx
	fill_array<<<block_num, thread_num >>>(xout, 1. / (1. * nx), nx);

	// compute the product of [I-Q]^(-1) with b
	for (int z = 0; z < n+1; z++) {
		cuda_q_vec(ptheta, xout, tmp, n, false, transp);
		add_arrays<<<block_num, thread_num >>>(b, tmp, nx);
		divide_arrays_elementwise<<<block_num, thread_num >>>(tmp, dg, xout, nx);
	}
}

/**
 * this functions shuffles the entries of old_vec into the entries of to_shuffle_vec
 *
 * @param[in] old_vec array that should be shuffled
 * @param[out] to_shuffle_vec array in which the shuffled vector is stored
 * @param[in] nx size of both vectors
*/
__global__ void shuffle(const double* __restrict__ old_vec, double* __restrict__ to_shuffle_vec, const int nx) {
	int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = cuda_index; k < nx; k += stride) {
		int greater_than_nx = (k >= nx / 2);
		to_shuffle_vec[k] = old_vec[2 * (k - greater_than_nx * nx / 2) + greater_than_nx];
	}
}


/**
 * inspired by https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * computes the sum of all entries in a given array
*/
__global__ void sum_over_array(const double* __restrict__ arr, double* __restrict__ result, int size) {

	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	double partial_sum = 0;

	for (unsigned int s = i; s < size; s += stride) {
		partial_sum += arr[s];
	}

	sdata[tid] = partial_sum;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) result[blockIdx.x] = sdata[0];
}


__global__ void print_vec(double *vec, int size) {
	int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = cuda_index; k < size; k+=stride) {
		printf("%g, ", vec[k]);
	}
	printf("\n\n");
}


__global__ void log_array(const double* __restrict__ input, double* __restrict__ output, int size){
	int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = cuda_index; k < size; k += stride) {
		output[k] = log(input[k]);
	}
}

/**
 * computes the marginal log-likelihood score given the relative frequency of observed tumours and the probability distribution yielded by the MHN
 *
 * @param[in] pD relative frequency of observed tumours in the data
 * @param[in] pth probability distribution yielded by the MHN
 * @param[in] n number of genes considered by the MHN
 * @param[out] score_out the marginal log-likelihood score is stored at this address
 * @param[in, out] tmp1 allocated memory of size 2^n needed by this function to operate
 * @param[in, out] tmp2 allocated memory of size >=1024 needed by this function to operate
*/
static void compute_score(const double* __restrict__ pD, const double* __restrict__ pth, int n, double* __restrict__ score_out, double* __restrict__ tmp1, double* __restrict__ tmp2){
	int nx = 1 << n;
	int block_num, thread_num;
	determine_block_thread_num(block_num, thread_num, n);

	log_array<<<block_num, thread_num>>>(pth, tmp1, nx);
	multiply_arrays_elementwise<<<block_num, thread_num>>>(pD, tmp1, nx);
	sum_over_array <<<block_num, thread_num, thread_num * sizeof(double) >>> (tmp1, tmp2, nx);
	sum_over_array <<<1, block_num, block_num * sizeof(double) >>> (tmp2, score_out, block_num);
}


void _compute_inverse(const double * __restrict__ theta, const int n, const double * __restrict__ dg, double * __restrict__ xout, bool transp);

/**
 * compute the gradient for a given relative frequency of observed tumours in the data
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[out] grad array that will contain the gradient at the end, size n*n
 * @param[in] pD relative frequency of observed tumours in the data, size 2^n
 * @param[in] pth memory buffer needed for this function, size 2^n
 * @param[in] q memory buffer needed for this function, size 2^n
 * @param[in] tmp1 memory buffer needed for this function, size 2^n
 * @param[in] tmp2 memory buffer needed for this function, size 2^n
 * @param[out] score the marginal log-likelihood score of the MHN will be stored here
*/
static void cuda_gradient_and_score_computation(const double* __restrict__ ptheta, const int n, double* __restrict__ grad, double* __restrict__ pD, double* __restrict__ pth, double* __restrict__ q, double* __restrict__ tmp1, double* __restrict__ tmp2, double* __restrict__ score) {

	const int nx = 1 << n;
	int block_num, thread_num;
	determine_block_thread_num(block_num, thread_num, n);

	// alias tmp1 and tmp2 for the first part of this function for better readability
	double* p0 = tmp1;
	double* dg = tmp2;

	// set all entries of p0 to zero, set the first entry to one
	cudaMemset(pth, 0, nx * sizeof(double));
	fill_array <<<1, 1>>> (pth, 1., 1);

	// compute the diagonal for the jacobi calls
	compute_jacobi_diagonal(ptheta, n, dg);

	// q is here only used as temporary memory, because the memory is not needed yet for anything else
	// cuda_jacobi(ptheta, p0, false, n, pth, q, dg);
	// cudaMemcpy(pth, p0, nx * sizeof(double), cudaMemcpyDeviceToDevice);
	_compute_inverse(ptheta, n, dg, pth, false);

	divide_arrays_elementwise<<<block_num, thread_num>>>(pD, pth, pD, nx);

	// here p0 is used as temporary memory, because we do not need its contents any longer
	// cuda_jacobi(ptheta, pD, true, n, q, p0, dg);
	cudaMemcpy(q, pD, nx * sizeof(double), cudaMemcpyDeviceToDevice);
	_compute_inverse(ptheta, n, dg, q, true);

	double *old_vec, *shuffled_vec, *swap_vec;

	multiply_arrays_elementwise<<<block_num, thread_num>>>(pth, pD, nx);
	compute_score(pD, pth, n, score, tmp1, tmp2);

	for (int i = 0; i < n; i++) {
		cudaMemset(tmp1, 0, nx * sizeof(double));

		cuda_kronvec<<<block_num, thread_num, n*sizeof(double)>>>(ptheta, i, pth, true, false, n, tmp1);
		
		// tmp1 contains the result of the call to cuda_restricted_kronvec above
		multiply_arrays_elementwise<<<block_num, thread_num>>>(q, tmp1, nx);

		old_vec = tmp1;
		shuffled_vec = tmp2;
		double *grad_i = grad + i * n;

		// use the shuffle trick for a more efficient computation of the gradient
		for (int j = 0; j < n; j++) {
			// confusion warning: the pD here has nothing to do with the former pD above
			// in this section pD is used again, because we need an allocated array and pD isnt needed anymore so we can just use that as memory
			shuffle<<<block_num, thread_num>>>(old_vec, shuffled_vec, nx);
			if (i == j) {
				sum_over_array <<<block_num, thread_num, thread_num * sizeof(double) >>> (shuffled_vec, pD, nx);
				sum_over_array <<<1, block_num, block_num * sizeof(double) >>> (pD, grad_i + i, block_num);
			}
			else {
				sum_over_array <<<block_num, thread_num, thread_num * sizeof(double) >>> (shuffled_vec + nx/2, pD, nx/2);
				sum_over_array <<<1, block_num, block_num * sizeof(double) >>> (pD, grad_i + j, block_num);
			}

			swap_vec = old_vec;
			old_vec = shuffled_vec;
			shuffled_vec = swap_vec;
		}
	}
}

__global__ void array_exp(double *arr, int size) {
	int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = cuda_index; i < size; i += stride) {
		arr[i] = exp(arr[i]);
	}
}


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
extern "C" int DLL_PREFIX cuda_full_state_space_gradient_score(double *ptheta, int n, double *pD, double *grad_out, double *score_out) {
	const int nx = 1 << n;

	double *cuda_grad_out;
	double *cuda_pD, *pth, *q, *tmp1, *tmp2;
	double *cuda_ptheta;
	double *cuda_score;

	// allocate memory on the GPU
	// we allocate all at once so that we can easily check for allocation errors
	// if we did each allocation as a separate cudaMalloc, we would have to check for errors after each single call
	double *d_memory;
	cudaMalloc(&d_memory,
				n*n * sizeof(double) +  // cuda_grad_out
				nx  * sizeof(double) +  // p0_pD
				nx  * sizeof(double) +  // pth
				nx  * sizeof(double) +  // q
				nx  * sizeof(double) +  // tmp1
				nx  * sizeof(double) +  // tmp2
				n*n * sizeof(double) +  // cuda_ptheta
				 1  * sizeof(double)    // cuda_score
				 );

	// check for errors
	// errors could occur if CUDA is not installed correctly or if the user tries to allocate too much memory
	if (cudaPeekAtLastError() != cudaSuccess){
		// cast cudaError_t to int, not the best style, but simplest method to get it work in Cython
		return (int) cudaPeekAtLastError();
	}

	cuda_grad_out = d_memory;
	cuda_pD       = d_memory + n*n;
	pth           = d_memory + n*n +   nx;
	q             = d_memory + n*n + 2*nx;
	tmp1          = d_memory + n*n + 3*nx;
	tmp2          = d_memory + n*n + 4*nx;
	cuda_ptheta   = d_memory + n*n + 5*nx;
	cuda_score    = d_memory + 2*n*n + 5*nx;

	// copy theta and pD to the GPU
	cudaMemcpy(cuda_ptheta, ptheta, n*n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_pD, pD, nx * sizeof(double), cudaMemcpyHostToDevice);

	// initialize the gradient on the GPU with zero
	cudaMemset(cuda_grad_out, 0, n*n * sizeof(double));

	// for the functions we need theta in its exponential form
	array_exp<<<32, 64>>>(cuda_ptheta, n*n);

	// again check for errors
	// errors could occur if CUDA is not installed correctly or the kernel call did not work correctly
	if (cudaPeekAtLastError() != cudaSuccess){
		// cast cudaError_t to int, not the best style, but simplest method to get it work in Cython
		return (int) cudaPeekAtLastError();
	}

	cuda_gradient_and_score_computation(cuda_ptheta, n, cuda_grad_out, cuda_pD, pth, q, tmp1, tmp2, cuda_score);

	// copy the results to the CPU
	cudaMemcpy(grad_out, cuda_grad_out, n*n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(score_out, cuda_score, sizeof(double), cudaMemcpyDeviceToHost);

	// free all memory on the GPU
	cudaFree(d_memory);

	return (int) cudaGetLastError();
}




// small function to compute n over k
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
 * this function represents a bijective mapping between the numbers 0 to (n over k) and the numbers which are smaller than 2^n and
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
	// (current_n-1 over k) for the subtree where we do not set the current bit to 1 and 
	// (current_n-1 over k-1) for the subtree where we do set the current bit to 1
	// if the given number i is greater than the size of the subtree where the bit is not set to 1, we set the bit to 1 and
	// subtract the size of that subtree from i for the next iteration

    for(int j = 0; j < n; j++){
		// compute (current_n-1 over k)
        binom_coef = ((current_n-k) * binom_coef) / current_n;
        if (i >= binom_coef) {
            index |= bit_setter;
			i -= binom_coef;
			// compute (current_n-1 over k-1)
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
 * @param[in, out] array containing the b at the beginning and x at the end
 * @param[in] j number of bits set to 1 in all indices for which the equation is solved
 * @param[in] binom_coef value of n over j
*/
__global__ void compute_inverse_level(const double * __restrict__ theta, const int n, const double * __restrict__ dg, double * __restrict__ xout, int j, int binom_coef){
	const int stride = blockDim.x * gridDim.x;
	const int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ double local_theta[];

	for(int i = threadIdx.x; i < n*n; i += blockDim.x){
		local_theta[i] = theta[i];
	}

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
 * @param[in, out] array containing the b at the beginning and x at the end
 * @param[in] j number of bits set to 1 in all indices for which the equation is solved
 * @param[in] binom_coef value of n over j
*/
__global__ void compute_inverse_level_t(const double * __restrict__ theta, const int n, const double * __restrict__ dg, double * __restrict__ xout, int j, int binom_coef){
	const int stride = blockDim.x * gridDim.x;
	const int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ double local_theta[];

	for(int i = threadIdx.x; i < n*n; i += blockDim.x){
		local_theta[i] = theta[i];
	}

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
void _compute_inverse(const double * __restrict__ theta, const int n, const double * __restrict__ dg, double * __restrict__ xout, bool transp = false){

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


/**
 * computes the solution for [I-Q] x = b using forward and backward substitution
 * 
 * @param[in] theta theta matrix representing the MHN with size n x n
 * @param[in] n number of rows and column of the theta matrix
 * @param[in] b vector of size 2^n which should be multiplied with [I-Q]^(-1)
 * @param[out] xout array of size 2^n which will contain the result of the matrix-vector multiplication at the end
*/
extern "C" void DLL_PREFIX gpu_compute_inverse(double *theta, int n, double *b, double *xout){


	int nx = 1 << n;
	int block_num, thread_num;

	determine_block_thread_num(block_num, thread_num, n);

	double *d_theta;
	double *d_b, *d_xout;
	double *d_dg;

	cudaMalloc(&d_theta, n*n * sizeof(double));
	cudaMalloc(&d_xout, nx * sizeof(double));
	cudaMalloc(&d_dg, nx * sizeof(double));

	cudaMemcpy(d_theta, theta, n*n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_xout, b, nx * sizeof(double), cudaMemcpyHostToDevice);

	array_exp<<<32, 64>>>(d_theta, n*n);

	fill_array<<<block_num, thread_num>>>(d_dg, 1, nx);
	cuda_subtract_q_diag(d_theta, n, d_dg, block_num, thread_num);

	_compute_inverse(d_theta, n, d_dg, d_xout, true);


	cudaMemcpy(xout, d_xout, nx * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_theta);
	cudaFree(d_xout);
	cudaFree(d_dg);
}
