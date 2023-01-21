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



void dua(cublasHandle_t handle, const double *theta, int n, double *bq, const State *state, double t, int i, int k, double eps, double gamma, double dgamma, double *pt, double *dp, double *dq, double *tmp, double *tmp2){

    int mutation_num = get_mutation_num(state);
    int nx = 1 << mutation_num;
    int nn = 0;

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
        cublasDaxpy(handle, nx, &gfac, q, 1, dp, 1);

        nn += 1;

        cuda_q_vec(theta, q, state, tmp, n, mutation_num, true, false);
        cublasDscal(handle, nx, &dgam_inv, tmp, 1);
        

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

        cudaMemcpy(cuda_theta, ptheta, n*n * sizeof(double), cudaMemcpyHostToDevice);


        for (int k = 1; k < data_size; k++){
            int current_mutation_num = get_mutation_num(&mutation_data[k]);
            int current_nx = 1 << current_mutation_num;

            determine_block_thread_num(block_num, thread_num, current_mutation_num);

            cudaMemset(bq, 0, current_nx * sizeof(double));

            int xk_index = empirical_distribution_index(&mutation_data[k], &mutation_data[k-1]);
            fill_array<<<1, 1>>>(bq + xk_index, 1.0, sizeof(double));

            double t = ages[k] - ages[k-1];

            double gamma = calc_gamma(handle, cuda_theta, n, &mutation_data[k], dg);

            for (int i = 0; i < n; i++){

                // compute the derivative of the diagonal using the shuffle trick
                cudaMemset(deriv_dg, 0, current_nx * sizeof(double));
                cuda_subdiag<<<block_num, thread_num>>>(theta, &mutation_data[k], i, n, current_mutation_num, deriv_dg);

                uint32_t state_copy = mutation_data[k]->parts[0];

                for(int j = 0; j < n; j++){

                    if (state_copy & 1){

                    } else if (i == j){

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

        cublasDestroy(handle);
    }

}