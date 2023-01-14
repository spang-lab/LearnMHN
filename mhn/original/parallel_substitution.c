
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include<stdio.h>

#ifndef NUMBER_OF_THREADS
#define NUMBER_OF_THREADS 2
#endif


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


int compute_index(int i, int n, int k){
    int index = 0;
    int bit_setter = 1;
    int separator = 0;
    int binom_coef;
    int current_n = n;

    for(int j = 0; j < n; j++){
        //binom_coef = ((current_n-k) * binom_coef) / current_n;
        binom_coef = compute_binom_coef(current_n-1, k);
        separator += binom_coef;

        if(i < separator){
            separator -= binom_coef;
        } else {
            index |= bit_setter;
            // binom_coef = (k * binom_coef) / (current_n - k);
            k -= 1;
            if(k == 0)
                break;
        }
        current_n -= 1;
        bit_setter <<= 1;
    }

    return index;
}


void _parallel_compute_inverse(const double * restrict theta, const int n, const double * restrict dg, const double * restrict b, double * restrict xout){

    int nx = 1 << n;
    double *exp_theta;
    int i, j, k, r;
    int binom_coef;
    int index, modified_index;
    int bit_setter;
    double theta_product, xout_element;
    int ind_copy;

    memcpy(xout, b, nx * sizeof(double));
    omp_set_num_threads(NUMBER_OF_THREADS);

    #pragma omp parallel shared(theta, dg, xout, n) private(exp_theta, i, j, k, r, index, modified_index, bit_setter, theta_product, xout_element, ind_copy)
    {
        exp_theta = malloc(n*n * sizeof(double));

        for(i = 0; i < n*n; i++){
            exp_theta[i] = exp(theta[i]);        
        }

        for(j = 0; j <= n; j++){
            binom_coef = compute_binom_coef(n, j);
            #pragma omp for
            for(i = 0; i < binom_coef; i++){
                index = compute_index(i, n, j);
                bit_setter = 1;
                xout_element = xout[index];
                for(k = 0; k < n; k++){
                    modified_index = (index & (~bit_setter));
                    if (modified_index != index){
                        theta_product = 1.;
                        ind_copy = index;
                        for(r = 0; r < n; r++){
                            if (ind_copy & 1){
                                theta_product *= exp_theta[k*n + r];
                            }
                            ind_copy >>= 1;
                        }
                        xout_element += theta_product * xout[modified_index];
                    }
                    bit_setter <<= 1;
                }
                xout[index] = xout_element / dg[index];
            }
        }
        free(exp_theta);
    }
}