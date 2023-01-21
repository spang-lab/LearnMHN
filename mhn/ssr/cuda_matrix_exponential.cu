// by Stefan Vocht
// this file contains the CUDA implementation of functions related to the matrix exponential used to compute 
// the scores and gradients for training a MHN on data where the sample ages are known


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_state_space_restriction.cuh"


void calc_gamma(const double *theta, const State *state, int i, int k, double *tmp1, double *tmp2){
    int mutation_num = get_mutation_num(state);
    int nx = 1 << mutation_num;


}



void dua(const double *theta, double *bq, const State *state, double t, int i, int k, double eps, double *pt, double *dp, double *dq, double *tmp, double *tmp2){

    int mutation_num = get_mutation_num(state);
    int nx = 1 << mutation_num;
    int n = 0;

    cudaMemset(pt, 0, nx * sizeof(double));
    cudaMemset(dp, 0, nx * sizeof(double));
    cudaMemset(dq, 0, nx * sizeof(double));


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

        double *bq, *dq;
        double *pt, *dp;
        double *tmp, *tmp2;
        double *cuda_grad, *cuda_score;
        double *cuda_theta;

        cudaMalloc(&bq, nx * sizeof(double));
        cudaMalloc(&pt, nx * sizeof(double));
        cudaMalloc(&dp, nx * sizeof(double));
        cudaMalloc(&dq, nx * sizeof(double));
        cudaMalloc(&tmp, nx * sizeof(double));
        cudaMalloc(&tmp2, nx * sizeof(double));
        cudaMalloc(&cuda_grad, n*n * sizeof(double));
        cudaMalloc(&cuda_score, sizeof(double));
        cudaMalloc(&cuda_theta, n*n * sizeof(double));


        cudaFree(bq);
        cudaFree(dq);
        cudaFree(pt);
        cudaFree(dp);
        cudaFree(tmp);
        cudaFree(tmp2);
        cudaFree(cuda_grad);
        cudaFree(cuda_score);
        cudaFree(cuda_theta);
    }

}