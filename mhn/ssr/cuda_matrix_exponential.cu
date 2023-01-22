// by Stefan Vocht
// this file contains the CUDA implementation of functions related to the matrix exponential used to compute 
// the scores and gradients for training a MHN on data where the sample ages are known


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cublas_v2.h>

#include <cmath>

#include "cuda_state_space_restriction.cuh"


double calc_gamma(cublasHandle_t handle, const double *theta, int n, const State *state, double *dg){
    int mutation_num = get_mutation_num(state);
    int nx = 1 << mutation_num;
    int block_num, thread_num;
    double gamma;
    determine_block_thread_num(block_num, thread_num);

    cudaMemset(dg, 0, nx * sizeof(double));
    cuda_subtract_q_diag(theta, state, n, mutation_num, dg, block_num, thread_num);
    cublasDnrm2(handle, nx, dg, 1, &gamma);
    return gamma;
}


// TODO optimize this later
__global__ void zero_mask(double *arr, int k, int size){
    int stride = blockDim.x * gridDim.x;
	int cuda_index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j = cuda_index; j < size; j += stride) {
		arr[k] = ((j >> k) & 1) * arr[k];
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

        cublasDaxpy(handle, nx, &ewg, bq, 1, pt, 1);

        cublasDaxpy(handle, nx, &ewg, dq, 1, dp, 1);

        gfac = ewg*dgamma*(nn/gamma - t);
        cublasDaxpy(handle, nx, &gfac, bq, 1, dp, 1);

        nn += 1;

        cuda_q_vec(theta, bq, state, tmp, n, mutation_num, true, false);
        cublasDscal(handle, nx, &dgam_inv, tmp, 1);
        cudaMemset(tmp2, 0, nx * sizeof(double));
        cuda_restricted_kronvec<<<block_num, thread_num, n * sizeof(double)>>>(theta, i, bq, state, true, false, n, mutation_num, count_before_i, tmp2);
        zero_mask<<<block_num, thread_num>>>(tmp2, k, nx);
        cublasDaxpy(handle, &gam_inv, tmp2, 1, tmp, 1);

        cuda_q_vec(theta, dq, state, tmp2, n, mutation_num, true, false);
        cublasDaxpy(handle, nx, &gam_inv, tmp2, 1, dq, 1);
        add_arrays<<<block_num, thread_num>>>(tmp, dq, nx);

        cuda_q_vec(theta, bq, state, tmp, n, mutation_num, true, false);
        cublasDaxpy(handle, nx, &gam_inv, tmp, 1, bq, 1);

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


        for (int k = 1; k < data_size; k++){
            int current_mutation_num = get_mutation_num(&mutation_data[k]);
            int current_nx = 1 << current_mutation_num;
            int current_nx_half = current_nx / 2;

            determine_block_thread_num(block_num, thread_num, current_mutation_num);

            cudaMemset(bq, 0, current_nx * sizeof(double));

            int xk_index = empirical_distribution_index(&mutation_data[k], &mutation_data[k-1]);
            fill_array<<<1, 1>>>(bq + xk_index, 1.0, sizeof(double));

            double t = ages[k] - ages[k-1];

            double gamma = calc_gamma(handle, cuda_theta, n, &mutation_data[k], dg);

            uint32_t state_copy_i = mutation_data[k]->parts[0];
            int count_before_i = 0;  // counts the number of mutations that occured before the ith index

            for (int i = 0; i < n; i++){

                // compute the derivative of the diagonal using the shuffle trick
                cudaMemset(deriv_dg, 0, current_nx * sizeof(double));
                cuda_subdiag<<<block_num, thread_num>>>(theta, &mutation_data[k], i, n, current_mutation_num, deriv_dg);
                multiply_arrays_elementwise<<<block_num, thread_num>>>(dg, deriv_dg, current_nx);

                uint32_t state_copy_j = mutation_data[k]->parts[0];

                for(int j = 0; j < n; j++){
                    if (state_copy_j & 1){
                        // shuffle deriv_dg
                        cublasDcopy(handle, current_nx_half, deriv_dg, 2, dq, 1);
                        cublasDcopy(handle, current_nx_half, deriv_dg+1, 2, dq + current_nx_half, 1);
                        cublasDcopy(handle, current_nx, dq, 1, deriv_dg, 1);   // this should be optimized by switching pointers

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
                        dua(handle, cuda_theta, n, bq, &mutation_data[k], t, i, j, eps, gamma, dgamma, pt, dp, dq, tmp, tmp2, count_before_i);
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
                        state_copy_j = state->parts[(j + 1) >> 5];
                    }
                }

                count_before_i += (state_copy_i & 1);

                if ((i + 1) & 31){
                     state_copy_i >>= 1;
                }
                else {
                    state_copy_i = state->parts[(i + 1) >> 5];
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

        cublasDestroy(handle);
    }

}