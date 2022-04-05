
// by Stefan Vocht
// this file contains the CUDA implementation of State Space Restrictions used to compute 
// the gradients for training a MHN


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <math.h>
#include <chrono>
#include <iostream>


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
     unsigned int parts[STATE_SIZE];
} State;


/**
 * Counts number of 1s in binary representation of number x, where x is a 32-bit integer
 * Source: https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
 *
 * @param[in] i the integer of which we want to count the number of set bits
*/
int count_ones32(uint32_t i){
	i = i - ((i >> 1) & 0x55555555);        			// add pairs of bits
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);  	// quads
    i = (i + (i >> 4)) & 0x0F0F0F0F;        			// groups of 8
    return (i * 0x01010101) >> 24;          			// horizontal sum of bytes
}


/**
 * Counts number of 1s in binary representation of number x, where x is a 64-bit integer
 * Source: https://en.wikipedia.org/wiki/Hamming_weight 
*
 * @param[in] i the int64 of which we want to count the number of set bits
*/
int count_ones(uint64_t x) {
    x -= (x >> 1) & 0x5555555555555555LL;             					//put count of each 2 bits into those 2 bits
    x = (x & 0x3333333333333333LL) + ((x >> 2) & 0x3333333333333333LL); //put count of each 4 bits into those 4 bits 
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fLL;        					//put count of each 8 bits into those 8 bits 
    return (x * 0x0101010101010101LL) >> 56;  							//returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ... 
}


// computes the number of mutations present in a given state
/**
 * Computes the number of mutations present in a given state
 *
 * @param[in] state A pointer to a State of which we want to count the number of mutations it contains
*/
int get_mutation_num(const State *state){
	int mutation_num = 0;
	for(int i = 0; i < STATE_SIZE; i++){
		mutation_num += count_ones32(state->parts[i]);
	}
	return mutation_num;
}


/**
 * we determine the number of blocks and threads used in the CUDA kernels for the current data point with this function
 *
 * @param[out] block_num number of blocks that should be used for the CUDA kernels
 * @param[out] thread_num number of threads that should be used for the CUDA kernels
 * @param[in] mutation_num number of mutations present in the current state
*/
inline void determine_block_thread_num(int &block_num, int &thread_num, const int mutation_num) {

	// block_num and thread_num have to be powers of two, else cuda_restricted_kronvec will not work
	// maximum 256 blocks with 1024 threads
	if (mutation_num >= 17) {
		block_num = 256;
		thread_num = 512;
	}
	// minimum 32 * STATE_SIZE threads, else for n = 32 * STATE_SIZE (which is the maximum possible n) not all thetas get loaded in kron_vec
	else if (mutation_num < 12) {
		block_num = 32;
		thread_num = 64;
	}
	else {
		block_num = 1 << (mutation_num / 2);
		thread_num = 1 << (mutation_num / 2 + (mutation_num & 1));
	}
}

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
__global__ void cuda_restricted_kronvec(const double* __restrict__ ptheta, const int i, const double* __restrict__ px, const State state, const bool diag, const bool transp, const int n, const int mutation_num, const int count_before_i, double* __restrict__ pout) {
	const int stride = blockDim.x * gridDim.x;
	const int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	// in the following 1 << i is equivalent to 2^i, x >> i is equivalent to x // 2^i, x & ((1<<i)-1) to x % 2^i
	const int nx = 1 << mutation_num;
	const int nxhalf = nx >> 1;

	extern __shared__ double theta_i[];

	// tells us, if state i is set to 1
	int state_i_one = (state.parts[i >> 5] >> (i & 31)) & 1;

	if (!diag && !state_i_one) {
		// in this case the result is zero in every entry
		// this means we do not have to add anything to pout and can just return
		return;
	}

	// load the ith row of theta into shared memory for more efficient access
	for(int j = threadIdx.x; j < n; j += blockDim.x){
		theta_i[j] = ptheta[i*n + j];
	}

	__syncthreads();


	// patch_size is important for later for the case i == j in the shuffle algorithm
	// as we do not actually shuffle the data in px in this implementation (only implicitly), we have to keep track of some indices
	// and which entries have to be computed together in the case i == j. Those two entries are px1 and px2
	// the index difference between those is patch_size (patches, as over all, the px1 and px2 occur in patches of size 2^z
	// z here is the number of events/bits set to 1 that have an index smaller than i)
	const int patch_size = 1 << count_before_i;
	int x_index1 = (cuda_index >> count_before_i) * 2 * patch_size + (cuda_index & (patch_size - 1));
	int x_index2 = x_index1 + patch_size;

	while(x_index2 < nx){
		double px1 = px[x_index1];
		double px2 = px[x_index2];

		int state_copy = state.parts[0];

		for (int j = 0; j < n; j++) {
			if (state_copy & 1) {
				// change the indices as they would change if we did an actual shuffle
				x_index1 = (x_index1 >> 1) + (x_index1 & 1) * nxhalf;
				x_index2 = (x_index2 >> 1) + (x_index2 & 1) * nxhalf;
				double theta = theta_i[j];

				if (i == j) {
					// at the beginning, x_index_1 and x_index_2 were chosen in such a way that for i == j those are the entries that are added/multiplied together
					// we want to know which entry is in which "column" (see original shuffle algorithm), they are loaded accordingly in right_side/left_side
					// here we prevent "if" diverging branches, which would lead to serial execution, by implicitly doing the if clause while computing right_side
					// as left_side must be the px that is not right_side, we get it with px1 + px2 - right_side (again no branching)
					double right_side, left_side;
					right_side = px1 + (x_index2 > x_index1) * (px2 - px1);
					left_side = px1 + px2 - right_side;

					// this part is pratically the same as in the original kronvec function
					if (!transp) {
						right_side = left_side * theta;
						if (diag) {
							left_side = -right_side;
						}
						else {
							left_side = 0;
						}
					}
					else {
						if (diag) {
							left_side = (right_side - left_side) * theta;
						}
						else {
							left_side = right_side * theta;
						}
						right_side = 0;
					}

					// update values of px1,px2 with the values from right_side/left_side (again with implicit if condition)
					px1 = right_side + (x_index1 < x_index2) * (left_side - right_side);
					px2 = right_side + left_side - px1;
				}
				else {
					// this is equivalent to "if(x_index >= nxhalf) px *= theta_i[j] else px *= 1
					// we dont use a if clause to prevent diverging branches
					px1 *= 1 + (x_index1 >= nxhalf) * (theta - 1);
					px2 *= 1 + (x_index2 >= nxhalf) * (theta - 1);
				}
			} 
			else if (i == j) {
				// if the ith gene is not mutated, we simply multiply the entries with (-theta_ii)
				px1 *= -theta_i[i];
				px2 *= -theta_i[i];
			}

			// if the mutation state of the next gene is stored on the current state_copy, make a bit shift to the right
			// else state_copy becomes the next integer stored in the given state (x >> 5  <=> x // 32, x & 31 <=> x % 32)
			if ((j + 1) & 31){
				state_copy >>= 1;
			}
			else {
				state_copy = state.parts[(j + 1) >> 5];
			}
			
		}
		// add the px values to the output array
		pout[x_index1] += px1;
		pout[x_index2] += px2;

		// if patch_size is bigger than stride, we have to do corrections to the indices
		if(stride < patch_size){
			// check if the current index is inside an odd patch, if so, jump to the next one
			x_index1 += stride;
			x_index1 += ((x_index1 >> count_before_i) & 1) * patch_size;
			x_index2 = x_index1 + patch_size;
		} else {
			x_index1 += 2*stride;
			x_index2 += 2*stride;
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
 * @param[in] state state representing current tumor sample
 * @param[out] yout array in which the result is stored
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[in] mutation_num number of mutations present in the current state / tumor sample
 * @param[in] diag if false, the diag of Q is set to zero during multiplication
 * @param[in] transp if true, multiplication is done with the transposed Q
*/
static void cuda_q_vec(const double *ptheta, const double *x, const State *state, double *yout, const int n, const int mutation_num, const bool diag, const bool transp) {
	
	const int nx = 1 << mutation_num;
	cudaMemset(yout, 0, nx * sizeof(double));

	int block_num, thread_num;
	int mutation_counter = 0;

	determine_block_thread_num(block_num, thread_num, mutation_num);

	for (int i = 0; i < n; i++) {
		// this would also be done in the kernel, but its faster to check it here
		if (!((state->parts[i >> 5] >> (i & 31)) & 1) && !diag) continue;

		cuda_restricted_kronvec<<<block_num, thread_num, n * sizeof(double)>>>(ptheta, i, x, *state, diag, transp, n, mutation_num, mutation_counter, yout);
		mutation_counter++;
	}
}


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
__global__ void cuda_subdiag(const double *ptheta, const State state, const int i, const int n, const int mutation_num, double *dg) {
	int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	// in the following 1 << i is equivalent to 2^i, x >> i is equivalent to x // 2^i, x & ((1<<i)-1) to x % 2^i
	const int nx = 1 << mutation_num;

	// store the ith row of theta in shared memory for more efficient access
	extern __shared__ double theta_i[];

	for(int j = threadIdx.x; j < n; j += blockDim.x){
		theta_i[j] = ptheta[i*n + j];
	}
	__syncthreads();

	
	for (int k = cuda_index; k < nx; k += stride) {

		double dg_entry = 1;

		int state_copy = state.parts[0];
		int position_condition = k;
		for (int j = 0; j < n; j++) {
			double theta = theta_i[j];
			// depending on the index different thetas have to be multiplied to the subdiag entry
			if (state_copy & 1) {
				if (i == j) {
					dg_entry *= -(1 - (position_condition & 1)) * theta;
				}
				else {
					dg_entry *= 1 + (position_condition & 1) * (theta - 1);
				}

				position_condition >>= 1;
			}
			else if (i == j) {
				dg_entry *= -theta;
			}

			// if the mutation state of the next gene is stored on the current state_copy, make a bit shift to the right
			// else state_copy becomes the next integer stored in the given state (x >> 5  <=> x // 32, x & 31 <=> x % 32)
			if ((j + 1) & 31){
				state_copy >>= 1;
			}
			else {
				state_copy = state.parts[(j + 1) >> 5];
			}
		}
		//subtract the subdiagonal from the diagonal entries
		dg[k] -= dg_entry;
	}
}



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
static void cuda_subtract_q_diag(const double *ptheta, const State *state, const int n, const int mutation_num, double *dg, int block_num, int thread_num) {
	for (int i = 0; i < n; i++) {
		cuda_subdiag<<<block_num, thread_num, n * sizeof(double)>>>(ptheta, *state, i, n, mutation_num, dg);
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
 * this functions multiplies [I-Q]^(-1) with b
 * all arrays given to this function must be allocated using cudaMalloc()
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] b array that is multiplied with [I-Q]^(-1)
 * @param[in] state state representing current tumor sample
 * @param[in] mutation_num number of mutations present in the current state / tumor sample
 * @param[in] transp if true, b is multiplied with the tranposed [I-Q]^(-1)
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[out] xout the results of this functio are stored in this array
 * @param[in, out] tmp this array is used to store temporary data, has to have size 2^mutation_num
 * @param[in, out] dg this array stores the diagonal of [I-Q], must have size 2^mutation_num
*/
static void cuda_jacobi(const double *ptheta, const double *b, const State *state, const int mutation_num, const bool transp, const int n, double *xout, double *tmp, double *dg) {

	const int nx = 1 << mutation_num;

	int block_num, thread_num;
	determine_block_thread_num(block_num, thread_num, mutation_num);

	// initialize the diagonal entries and xout
	fill_array<<<block_num, thread_num >>>(dg, 1, nx);
	cuda_subtract_q_diag(ptheta, state, n, mutation_num, dg, block_num, thread_num);
	fill_array<<<block_num, thread_num >>>(xout, 1. / (1. * nx), nx);

	// compute the product of [I-Q]^(-1) with b
	for (int z = 0; z < mutation_num + 1; z++) {
		cuda_q_vec(ptheta, xout, state, tmp, n, mutation_num, false, transp);
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
__global__ void sum_over_array(const double *arr, double *result, int size) {

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

	for (int k = cuda_index; k < size; k++) {
		printf("%g, ", vec[k]);
	}
	printf("\n\n");
}


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
static void cuda_restricted_gradient(const double *ptheta, const State *state, const int n, double *grad, double *p0_pD, double *pth, double *q, double *tmp1, double *tmp2) {

	// get the number of mutated genes in the current sample and compute the size of the memory buffers
	const int mutation_num = get_mutation_num(state);
	const int nx = 1 << mutation_num;

	const double one = 1;

	// set all entries of p0_pD to zero, set the first entry to one
	cudaMemset(p0_pD, 0, nx * sizeof(double));
	cudaMemcpy(p0_pD, &one, sizeof(double), cudaMemcpyHostToDevice);

	cuda_jacobi(ptheta, p0_pD, state, mutation_num, false, n, pth, tmp1, tmp2);

	// set all entries of p0_pD to zero, set the last entry to 1/pth[last_index]
	cudaMemset(p0_pD, 0, sizeof(double));
	cudaMemcpy(p0_pD + nx - 1, &one, sizeof(double), cudaMemcpyHostToDevice);
	divide_arrays_elementwise<<<1, 1>>>(p0_pD + nx - 1, pth + nx - 1, p0_pD + nx - 1, 1);

	cuda_jacobi(ptheta, p0_pD, state, mutation_num, true, n, q, tmp1, tmp2);

	double *old_vec, *shuffled_vec, *swap_vec;
	int block_num, thread_num;

	determine_block_thread_num(block_num, thread_num, mutation_num);

	// initialize grad with zeros
	cudaMemset(grad, 0, n*n * sizeof(double));

	// this counter is used for cuda_restricted_kronvec and counts how many of the genes
	// up to this point have been mutated in the tumor sample
	int kronvec_count_before_i = 0;

	for (int i = 0; i < n; i++) {
		cudaMemset(tmp1, 0, nx * sizeof(double));

		// check if the current gene is mutated
		if((state->parts[i >> 5] >> (i & 31)) & 1){
			cuda_restricted_kronvec<<<block_num, thread_num, n*sizeof(double)>>>(ptheta, i, pth, *state, true, false, n, mutation_num, kronvec_count_before_i, tmp1);
			kronvec_count_before_i++;
		}
		else{
			// if the current gene is not mutated we can set the parameter count_before_i to mutation_num-1 to get a patch_size as large as possible
			// this gives us better aligned memory access on global memory
			cuda_restricted_kronvec<<<block_num, thread_num, n*sizeof(double)>>>(ptheta, i, pth, *state, true, false, n, mutation_num, mutation_num - 1, tmp1);
		}

		// tmp1 contains the result of the call to cuda_restricted_kronvec above
		multiply_arrays_elementwise<<<block_num, thread_num>>>(q, tmp1, nx);

		old_vec = tmp1;
		shuffled_vec = tmp2;
		int state_copy = state->parts[0];
		double *grad_i = grad + i * n;

		// use the shuffle trick for a more efficient computation of the gradient
		for (int j = 0; j < n; j++) {
			// confusion warning: the p0_pD here has nothing to do with p0 or pD
			// in this section p0_pD is used again, because we need an allocated array and p0_pD isnt needed anymore so we can just use that as memory
			if (state_copy & 1) {
				shuffle<<<block_num, thread_num>>>(old_vec, shuffled_vec, nx);
				if (i == j) {
					sum_over_array <<<block_num, thread_num, thread_num * sizeof(double) >>> (shuffled_vec, p0_pD, nx);
					sum_over_array <<<1, block_num, block_num * sizeof(double) >>> (p0_pD, grad_i + i, block_num);
				}
				else {
					sum_over_array <<<block_num, thread_num, thread_num * sizeof(double) >>> (shuffled_vec + nx/2, p0_pD, nx/2);
					sum_over_array <<<1, block_num, block_num * sizeof(double) >>> (p0_pD, grad_i + j, block_num);
				}

				swap_vec = old_vec;
				old_vec = shuffled_vec;
				shuffled_vec = swap_vec;
			}
			else if (i == j) {
				sum_over_array<<<block_num, thread_num, thread_num * sizeof(double)>>>(old_vec, p0_pD, nx);
				sum_over_array<<<1, block_num, block_num * sizeof(double)>>>(p0_pD, grad_i + i, block_num);
			}

			// if the mutation state of the next gene is stored on the current state_copy, make a bit shift to the right
			// else state_copy becomes the next integer stored in the given state (x >> 5  <=> x // 32, x & 31 <=> x % 32)
			if ((j + 1) & 31){
				state_copy >>= 1;
			}
			else {
				state_copy = state->parts[(j + 1) >> 5];
			}
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

__global__ void add_to_score(double *score, double *pth_end){
	const int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(cuda_index == 0){
		score[0] += log(pth_end[0]);
	}
}


/**
 * this function computes the gradient and score for the current MHN for a given data set using CUDA
 *
 * @param[in] ptheta array containing the theta entries
 * @param[in] n number of genes considered by the MHN, also number of columns/rows of theta
 * @param[in] mutation_data array of States, where each state represents a tumor sample
 * @param[in] data_size number of tumor samples in mutation_data
 * @param[out] grad_out array of size n*n in which the gradient will be stored
 *
 * @return this function returns the score of the current MHN as d double value
*/
double DLL_PREFIX cuda_gradient_and_score(double *ptheta, int n, State *mutation_data, int data_size, double *grad_out) {

	// determine the maximum number of mutations present in a single tumor sample
	int max_mutation_num = 0;
	for (int i = 0; i < data_size; i++) {
		if (get_mutation_num(&mutation_data[i]) > max_mutation_num) max_mutation_num = get_mutation_num(&mutation_data[i]);
	}

	const int nx = 1 << max_mutation_num;

	double *cuda_grad_out, *partial_grad;
	double *p0_pD, *pth, *q, *tmp1, *tmp2;
	double *cuda_ptheta;
	double *cuda_score, score;

	// allocate memory on the GPU
	cudaMalloc(&cuda_grad_out, n*n * sizeof(double));
	cudaMalloc(&partial_grad, n*n * sizeof(double));
	cudaMalloc(&p0_pD, nx * sizeof(double));
	cudaMalloc(&pth, nx * sizeof(double));
	cudaMalloc(&q, nx * sizeof(double));
	cudaMalloc(&tmp1, nx * sizeof(double));
	cudaMalloc(&tmp2, nx * sizeof(double));
	cudaMalloc(&cuda_ptheta, n*n * sizeof(double));

	cudaMalloc(&cuda_score, sizeof(double));

	// copy theta to the GPU
	cudaMemcpy(cuda_ptheta, ptheta, n*n * sizeof(double), cudaMemcpyHostToDevice);
	
	// initialize the gradient on the GPU with zero
	cudaMemset(cuda_grad_out, 0, n*n * sizeof(double));

	// for the functions we need theta in its exponential form
	array_exp<<<32, 64>>>(cuda_ptheta, n*n);

	// compute the gradient for each tumor sample and add them together
	for (int i = 0; i < data_size; i++) {
		cuda_restricted_gradient(cuda_ptheta, &mutation_data[i], n, partial_grad, p0_pD, pth, q, tmp1, tmp2);
		add_arrays<<<32, 64>>>(partial_grad, cuda_grad_out, n*n);

		int mutation_num = get_mutation_num(&mutation_data[i]);
		add_to_score<<<1, 1>>>(cuda_score, &pth[(1 << mutation_num) - 1]);

		cudaDeviceSynchronize();
	}

	// copy the results to the CPU
	cudaMemcpy(grad_out, cuda_grad_out, n*n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&score, cuda_score, sizeof(double), cudaMemcpyDeviceToHost);

	// free all memory on the GPU
	cudaFree(partial_grad);
	cudaFree(p0_pD);
	cudaFree(pth);
	cudaFree(q);
	cudaFree(tmp1);
	cudaFree(tmp2);
	cudaFree(cuda_ptheta);

	cudaFree(cuda_score);
	cudaFree(cuda_grad_out);

	return score;
}
