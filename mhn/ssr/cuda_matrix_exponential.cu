// by Stefan Vocht
// this file contains the CUDA implementation of functions related to the matrix exponential used to compute 
// the scores and gradients for training a MHN on data where the sample ages are known


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

// #include <cublas_v2.h>

#include <cmath>

// not the most elegant way, but gets the job done
#include "cuda_state_space_restriction.cu"


// currently, cublas is not working with Cython on Windows
// so I wrote my own functions replacing the BLAS ones
// if cublas should work in the future, one can simply use the cublas calls that are commented out right now
// as for now, I aliased cublasHandle_t with int so that the code works without cublas
#ifndef CUBLAS_H_
#define cublasHandle_t int
#define cublasCreate(handle) *handle = 1
#endif


__global__ void myDaxpy(int size, double alpha, const double *x, double *y){
    int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = cuda_index; i < size; i += stride){
        y[i] += alpha * x[i];
    }
}


__global__ void myDscal(int size, double alpha, double *x){
    int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = cuda_index; i < size; i += stride){
        x[i] *= alpha;
    }    
}


/**
 * inspired by https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * computes part of the nrm2 for a given array
*/
__global__ void DLL_PREFIX _compute_partial_dnrm2(const double *arr, double *result, int size) {

	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	double partial_sum = 0;
    double current_val;

	for (unsigned int s = i; s < size; s += stride) {
        current_val = arr[s];
		partial_sum += (current_val * current_val);
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


double myDnrm2(int size, double *x, int block_num, int thread_num){
    double *buffer;
    double nrm2;
    cudaMalloc(&buffer, block_num * sizeof(double));
    _compute_partial_dnrm2<<<block_num, thread_num, thread_num * sizeof(double)>>>(x, buffer, size);
    sum_over_array<<<1, block_num, block_num * sizeof(double)>>>(buffer, buffer, block_num);
    cudaMemcpy(&nrm2, buffer, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(buffer);
    return sqrt(nrm2);
}


double calc_gamma(cublasHandle_t handle, const double *theta, int n, const State *state, double *dg){
    int mutation_num = get_mutation_num(state);
    int nx = 1 << mutation_num;
    int block_num, thread_num;
    double gamma;
    determine_block_thread_num(block_num, thread_num, mutation_num);

    cudaMemset(dg, 0, nx * sizeof(double));
    cuda_subtract_q_diag(theta, state, n, mutation_num, dg, block_num, thread_num);
    // cublasDnrm2(handle, nx, dg, 1, &gamma);
    gamma = myDnrm2(nx, dg, block_num, thread_num);
    return gamma;
}


// TODO optimize this later
__global__ void zero_mask(double *arr, int k, int size){
    int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j = cuda_index; j < size; j += stride) {
		arr[j] = ((j >> k) & 1) * arr[j];
	}
}


void dua(cublasHandle_t handle, const double *theta, int n, double *bq, const State *state, double t, int i, int k, double eps, double gamma, double dgamma, double *pt, double *dp, double *dq, double *tmp, double *tmp2, int count_before_i){

    int mutation_num = get_mutation_num(state);
    int nx = 1 << mutation_num;
    int nn = 0;

    int block_num, thread_num;
    determine_block_thread_num(block_num, thread_num, mutation_num);

    cudaMemset(pt, 0, nx * sizeof(double));
    cudaMemset(dp, 0, nx * sizeof(double));
    cudaMemset(dq, 0, nx * sizeof(double));

    double gfac = 1.;
    double dgam_inv = -1./(gamma*gamma) * dgamma;
    double gam_inv = 1 / gamma;
    double ewg = exp(-1. * gamma * t);
    double mass_defect = 0.0;

    while (eps < (1 - mass_defect)){
        mass_defect += ewg;

        // cublasDaxpy(handle, nx, &ewg, bq, 1, pt, 1);
        myDaxpy<<<block_num, thread_num>>>(nx, ewg, bq, pt);

        // cublasDaxpy(handle, nx, &ewg, dq, 1, dp, 1);
        myDaxpy<<<block_num, thread_num>>>(nx, ewg, dq, dp);

        gfac = ewg*dgamma*(nn/gamma - t);
        // cublasDaxpy(handle, nx, &gfac, bq, 1, dp, 1);
        myDaxpy<<<block_num, thread_num>>>(nx, gfac, bq, dp);

        nn += 1;

        cuda_q_vec(theta, bq, state, tmp, n, mutation_num, true, false);
        // cublasDscal(handle, nx, &dgam_inv, tmp, 1);
        myDscal<<<block_num, thread_num>>>(nx, dgam_inv, tmp);
        cudaMemset(tmp2, 0, nx * sizeof(double));
        cuda_restricted_kronvec<<<block_num, thread_num, n * sizeof(double)>>>(theta, i, bq, *state, true, false, n, mutation_num, count_before_i, tmp2);
        if (i != k){
            zero_mask<<<block_num, thread_num>>>(tmp2, k, nx);
        }  
        // cublasDaxpy(handle, nx, &gam_inv, tmp2, 1, tmp, 1);
        myDaxpy<<<block_num, thread_num>>>(nx, gam_inv, tmp2, tmp);

        cuda_q_vec(theta, dq, state, tmp2, n, mutation_num, true, false);
        // cublasDaxpy(handle, nx, &gam_inv, tmp2, 1, dq, 1);
        myDaxpy<<<block_num, thread_num>>>(nx, gam_inv, tmp2, dq);
        add_arrays<<<block_num, thread_num>>>(tmp, dq, nx);

        cuda_q_vec(theta, bq, state, tmp, n, mutation_num, true, false);
        // cublasDaxpy(handle, nx, &gam_inv, tmp, 1, bq, 1);
        myDaxpy<<<block_num, thread_num>>>(nx, gam_inv, tmp, bq);

        ewg *= gamma*t / nn;
    }
}


int empirical_distribution_index(const State *current_state, const State *former_state){
    int current_mutation_num = get_mutation_num(current_state);
    int xk_index = (1 << current_mutation_num) - 1;
    int bit_setter = 1;

    uint32_t state_copy_current = current_state->parts[0];
    uint32_t state_copy_former = former_state->parts[0];

    for(int j = 0; j < 32 * STATE_SIZE; j++){
        if (state_copy_current & 1){
            if (!(state_copy_former & 1)){
                xk_index &= ~bit_setter;
            }
            bit_setter <<= 1;
        }
        if ((j+1) & 31){
            state_copy_current >>= 1;
            state_copy_former >>= 1;
        } else {
            state_copy_current = current_state->parts[(j+1) >> 5];
            state_copy_former = former_state->parts[(j+1) >> 5];
        }
    }
    return xk_index;
}


extern "C" 
{
    int DLL_PREFIX cuda_gradient_and_score_dua(const double *ptheta, int n, const State *mutation_data, const double *ages, int data_size, double eps, double *grad_out, double *score_out){
        
        int max_mutation_num = 0;
        for (int i = 0; i < data_size; i++) {
            if (get_mutation_num(&mutation_data[i]) > max_mutation_num) max_mutation_num = get_mutation_num(&mutation_data[i]);
        }

        int nx = 1 << max_mutation_num;

        
        cublasHandle_t handle;
        cublasCreate(&handle);

        double *bq, *dq;
        double *pt, *dp;
        double *tmp, *tmp2;
        double *cuda_grad, *cuda_score;
        double *cuda_theta;
        double *dg, *deriv_dg;
        double *cuda_dgamma;
        double dgamma;

        int block_num, thread_num;

        cudaMalloc(&bq, nx * sizeof(double));
        cudaMalloc(&pt, nx * sizeof(double));
        cudaMalloc(&dp, nx * sizeof(double));
        cudaMalloc(&dq, nx * sizeof(double));
        cudaMalloc(&dg, nx * sizeof(double));
        cudaMalloc(&deriv_dg, nx * sizeof(double));
        cudaMalloc(&tmp, nx * sizeof(double));
        cudaMalloc(&tmp2, nx * sizeof(double));
        cudaMalloc(&cuda_grad, n*n * sizeof(double));
        cudaMalloc(&cuda_score, sizeof(double));
        cudaMalloc(&cuda_theta, n*n * sizeof(double));
        cudaMalloc(&cuda_dgamma, sizeof(double));

        cudaMemcpy(cuda_theta, ptheta, n*n * sizeof(double), cudaMemcpyHostToDevice);

        // for the functions we need theta in its exponential form
        array_exp<<<32, 64>>>(cuda_theta, n*n);

        cudaMemset(cuda_score, 0, sizeof(double));
        cudaMemset(cuda_grad, 0, n*n * sizeof(double));


        for (int k = 1; k < data_size; k++){
            const State *current_state = &mutation_data[k];
            int current_mutation_num = get_mutation_num(current_state);
            int current_nx = 1 << current_mutation_num;
            int current_nx_half = current_nx / 2;

            determine_block_thread_num(block_num, thread_num, current_mutation_num);

            int xk_index = empirical_distribution_index(current_state, &mutation_data[k-1]);

            double t = ages[k] - ages[k-1];

            double gamma = calc_gamma(handle, cuda_theta, n, current_state, dg);

            uint32_t state_copy_i = current_state->parts[0];
            int count_before_i = 0;  // counts the number of mutations that occured before the ith index

            for (int i = 0; i < n; i++){

                // compute the derivative of the diagonal using the shuffle trick
                cudaMemset(deriv_dg, 0, current_nx * sizeof(double));
                cuda_subdiag<<<block_num, thread_num>>>(cuda_theta, *current_state, i, n, current_mutation_num, deriv_dg);
                multiply_arrays_elementwise<<<block_num, thread_num>>>(dg, deriv_dg, current_nx);

                uint32_t state_copy_j = current_state->parts[0];

                for(int j = 0; j < n; j++){
                    if (state_copy_j & 1){
                        // shuffle deriv_dg
                        // cublasDcopy(handle, current_nx_half, deriv_dg, 2, dq, 1);
                        // cublasDcopy(handle, current_nx_half, deriv_dg+1, 2, dq + current_nx_half, 1);
                        // cublasDcopy(handle, current_nx, dq, 1, deriv_dg, 1);   // this should be optimized by switching pointers

                        shuffle<<<block_num, thread_num>>>(deriv_dg, dq, current_nx);
                        cudaMemcpy(deriv_dg, dq, current_nx * sizeof(double), cudaMemcpyDeviceToDevice);

                        if (i == j){
                            sum_over_array<<<block_num, thread_num, thread_num * sizeof(double)>>>(deriv_dg, dq, current_nx);
                            sum_over_array<<<1, block_num, block_num * sizeof(double)>>>(dq, cuda_dgamma, block_num);
                            cudaMemcpy(&dgamma, cuda_dgamma, sizeof(double), cudaMemcpyDeviceToHost);
                        } else {
                            sum_over_array<<<block_num, thread_num, thread_num * sizeof(double)>>>(deriv_dg + current_nx_half, dq, current_nx_half);
                            sum_over_array<<<1, block_num, block_num * sizeof(double)>>>(dq, cuda_dgamma, block_num);
                            cudaMemcpy(&dgamma, cuda_dgamma, sizeof(double), cudaMemcpyDeviceToHost);
                        }

                    } else if (i == j){
                        sum_over_array<<<block_num, thread_num, thread_num * sizeof(double)>>>(deriv_dg, dq, current_nx);
                        sum_over_array<<<1, block_num, block_num * sizeof(double)>>>(dq, cuda_dgamma, block_num);
                        cudaMemcpy(&dgamma, cuda_dgamma, sizeof(double), cudaMemcpyDeviceToHost);
                    }

                    if ((state_copy_j & 1) || i == j){
                        cudaMemset(bq, 0, current_nx * sizeof(double));
                        fill_array<<<1, 1>>>(bq + xk_index, 1.0, 1);
                        dgamma /= gamma;
                        dua(handle, cuda_theta, n, bq, current_state, t, i, j, eps, gamma, dgamma, pt, dp, dq, tmp, tmp2, count_before_i);
                        // add result to gradient
                        divide_arrays_elementwise<<<1, 1>>>(dp + current_nx - 1, pt + current_nx - 1, dp + current_nx - 1, 1);
                        add_arrays<<<1, 1>>>(dp + current_nx - 1, cuda_grad + i*n + j, 1);
                    }

                    // if the mutation state of the next gene is stored on the current state_copy_j, make a bit shift to the right
                    // else state_copy_j becomes the next integer stored in the given state (x >> 5  <=> x // 32, x & 31 <=> x % 32)
                    if ((j + 1) & 31){
                        state_copy_j >>= 1;
                    }
                    else {
                        state_copy_j = current_state->parts[(j + 1) >> 5];
                    }
                }

                count_before_i += (state_copy_i & 1);

                if ((i + 1) & 31){
                     state_copy_i >>= 1;
                }
                else {
                    state_copy_i = current_state->parts[(i + 1) >> 5];
                }
            }
            // update total score
            add_to_score<<<1, 1>>>(cuda_score, pt + current_nx - 1);
        }

        cudaMemcpy(grad_out, cuda_grad, n*n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(score_out, cuda_score, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(bq);
        cudaFree(dq);
        cudaFree(dg);
        cudaFree(deriv_dg);
        cudaFree(pt);
        cudaFree(dp);
        cudaFree(tmp);
        cudaFree(tmp2);
        cudaFree(cuda_grad);
        cudaFree(cuda_score);
        cudaFree(cuda_theta);
        cudaFree(cuda_dgamma);

        // cublasDestroy(handle);

        return (int) cudaGetLastError();
    }

}