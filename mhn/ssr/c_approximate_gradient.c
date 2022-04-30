#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>


// by Stefan Vocht
// here the functions of approximate_gradient_cython are implemented in C
// as the functions are practically the same there is not much documentation
// see the Cython version for more documentation


#ifdef __cplusplus
extern "C" {
#endif

/*typedef struct {
    unsigned int parts[STATE_SIZE];
} State;*/

// set the seed for random generator
void set_c_seed(int seed){
    srand(seed);
}


inline double q_next(double *theta, int n, int *curr_sequence, int curr_seq_len, int new_element){

    double result = 1;

    for(int i = 0; i < curr_seq_len; i++){
        result *= theta[new_element*n + curr_sequence[i]];
    }
    result *= theta[new_element*n + new_element];
    return result;
}


void q_next_deriv(double *theta, int n, int *curr_sequence, int curr_seq_len, int new_element, int i, double *rout){

    for(int j = 0; j < n; j++){
        rout[j] = 0;
    }

    if (i != new_element) return;

    double q_n = q_next(theta, n, curr_sequence, curr_seq_len, new_element);
    rout[i] = q_n;

    for(int j = 0; j < curr_seq_len; j++){
        rout[curr_sequence[j]] = q_n;
    }
}


// mem_buffer must be of size n
double q_tilde(double *theta, int n, int *sequence, int seq_len, int *mem_buffer){

    double result = 0;
    double r_loc;

    // aliasing mem_buffer for better readability
    int *in_sequence = mem_buffer;

    for(int i = 0; i < n; i++){
        in_sequence[i] = 0;
    }

    for(int i = 0; i < seq_len; i++){
        in_sequence[sequence[i]] = 1;
    }

    for(int i = 0; i < n; i++){
        if(in_sequence[i]) continue;

        r_loc = theta[i*n + i];
        for(int j = 0; j < seq_len; j++){
            r_loc *= theta[i*n + sequence[j]];
        }
        result += r_loc;
    }
    return result;
}


void q_tilde_deriv(double *theta, int n, int *sequence, int seq_len, int k, double *rout){

    for(int j = 0; j < n; j++){
        rout[j] = 0;
    }

    for(int j = 0; j < seq_len; j++){
        if(sequence[j] == k) return;
    }

    double q_n = q_next(theta, n, sequence, seq_len, k);

    rout[k] = q_n;
    for(int j = 0; j < seq_len; j++){
        rout[sequence[j]] = q_n;
    }
}


// mem_buffer must be of size n
double p_sigma(double *theta, int n, int *sequence, int seq_len, int *mem_buffer){

    double result = 1;
    double denominator = 1;

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j <= i; j++){
            result *= theta[sequence[i]*n + sequence[j]];
        }
        // result *= q_next(theta, n, sequence, i, sequence[i]);
    }

    for(int i = 0; i <= seq_len; i++){
        denominator *= 1 + q_tilde(theta, n, sequence, i, mem_buffer);
    }

    result /= denominator;
    return result;
}


void p_sigma_deriv(double *theta, int n, int *sequence, int seq_len, int k, double *rout){

    double p_loc, one_plus_tilde, q_n;

    for(int j = 0; j < n; j++){
        rout[j] = 0;
    }

    int *mem_buffer = (int *) malloc(n * sizeof(int));
    double *q_next_der = (double *) malloc(n * sizeof(double));
    double *q_tilde_der = (double *) malloc(n * sizeof(double));

    double p_sig = p_sigma(theta, n, sequence, seq_len, mem_buffer);

    for(int i = 0; i < seq_len; i++){
        p_loc = p_sig;
        one_plus_tilde = 1 + q_tilde(theta, n, sequence, i, mem_buffer);
        q_n = q_next(theta, n, sequence, i, sequence[i]);
        p_loc /= (one_plus_tilde * q_n);

        q_next_deriv(theta, n, sequence, i, sequence[i], k, q_next_der);
        q_tilde_deriv(theta, n, sequence, i, k, q_tilde_der);

        for(int j = 0; j < n; j++){
            rout[j] += p_loc * (one_plus_tilde * q_next_der[j] - q_n * q_tilde_der[j]);
        }
    }

    one_plus_tilde = 1 + q_tilde(theta, n, sequence, seq_len, mem_buffer);
    p_sig /= one_plus_tilde;
    q_tilde_deriv(theta, n, sequence, seq_len, k, q_tilde_der);

    for(int j = 0; j < n; j++){
        rout[j] -= p_sig * q_tilde_der[j];
    }

    free(mem_buffer);
    free(q_next_der);
    free(q_tilde_der);
}


double draw_from_q(double *theta, int n, int *s, int s_size, int *sigma_out){

    double q_val = 1;
    int v, new_entry_index = 0;
    bool in_sigma;
    double dv, tmp_sum, sum_u, random_num;

    int *s_without_sigma = (int *) malloc(s_size * sizeof(int));
    double *u = (double *) malloc(s_size * sizeof(double));

    for(int k = 0; k < s_size; k++){
        s_without_sigma[k] = s[k];
    }

    for(int k = 0; k < s_size; k++){
        for(int i = 0; i < s_size - k; i++){
            v = s_without_sigma[i];
            sigma_out[k] = v;
            dv = 1;
            for(int j = 0; j < n; j++){
                in_sigma = false;
                for(int t = 0; t <= k; t++){
                    if(sigma_out[t] == j){
                        in_sigma = true;
                        break;
                    }
                }
                if(in_sigma) continue;

                tmp_sum = theta[n*j + j];
                for(int t = 0; t <= k; t++){
                    tmp_sum *= theta[n*j + sigma_out[t]];
                }
                dv += tmp_sum;
            }

            tmp_sum = 1. / theta[v * n + v];
            for(int t = 0; t < s_size - k; t++){
                tmp_sum *= theta[n * s_without_sigma[t] + v];
            }
            u[i] = tmp_sum / dv;
        }

        sum_u = 0;
        for(int t = 0; t < s_size - k; t++){
            sum_u += u[t];
        }

        random_num = rand() / (1.0 * RAND_MAX) * sum_u;
        tmp_sum = 0;
        for(int t = 0; t < s_size - k; t++){
            tmp_sum += u[t];
            if (random_num <= tmp_sum){
                new_entry_index = t;
                break;
            }
        }
        sigma_out[k] = s_without_sigma[new_entry_index];
        q_val *= u[new_entry_index] / sum_u;

        for(int i = new_entry_index; i < s_size - k; i++){
            s_without_sigma[i] = s_without_sigma[i+1];
        }
    }

    free(s_without_sigma);
    free(u);

    return q_val;
}


/**
 * This function approximates the score and its gradient for the current MHN for a given tumor sample
 *
 * @param[in] theta array containing the theta entries
 * @param[in] n total number of genes considered by the MHN, also column and row size of theta
 * @param[in] state current state used to compute the gradient
 * @param[in] m number of paths sampled to compute the approximation
 * @param[in] burn_in_samples number of burn-in samples used before starting the approximation
 * @param[out] grad_out array which will contain the resulting gradient
 *
 * @return score corresponding to the given tumor sample
 */
double approximate_gradient_and_score(double *theta, int n, State *state, int m, int burn_in_samples, double *grad_out){

    double p_old, inv_p_old, p_new, p_accept;
    double q_val_old, q_val_new;
    int *sigma_old, *sigma_new, *tmp;
    double score = 0;

    int *mem_buffer = (int *) malloc(n * sizeof(int));

    bool is_zero = true;

    for(int i = 0; i < STATE_SIZE; i++){
        if(state->parts[i] != 0){
            is_zero = false;
            break;
        }
    }

    if(is_zero){
        inv_p_old = 1 / p_sigma(theta, n, (int *) NULL, 0, mem_buffer);
        for(int k = 0; k < n; k++){
            p_sigma_deriv(theta, n, (int *) NULL, 0, k, grad_out + k*n);
            for(int j = 0; j < n; j++){
                grad_out[k*n + j] *= inv_p_old;
            }
        }
        free(mem_buffer);
        // in this case the score is just the log of p_sigma
        return -log(inv_p_old);
    }

    for(int i = 0; i < n*n; i++){
        grad_out[i] = 0;
    }

    int *state_as_array = (int *) malloc(n * sizeof(int));
    int mutation_num = 0;

    for(int i = 0; i < n; i++){
        if((state->parts[i >> 5] >> (i & 31)) & 1){
            state_as_array[mutation_num] = i;
            mutation_num++;
        }
    }

    sigma_old = (int *) malloc(n * sizeof(int));
    sigma_new = (int *) malloc(n * sizeof(int));

    q_val_old = draw_from_q(theta, n, state_as_array, mutation_num, sigma_old);
    p_old = p_sigma(theta, n, sigma_old, mutation_num, mem_buffer);

    for(int i = 0; i < burn_in_samples; i++){
        q_val_new = draw_from_q(theta, n, state_as_array, mutation_num, sigma_new);
        p_new = p_sigma(theta, n, sigma_new, mutation_num, mem_buffer);
        p_accept = (p_new * q_val_old) / (p_old * q_val_new);

        if (p_accept > 1){
            p_accept = 1;
        }

        if (rand() <= p_accept * RAND_MAX){
            p_old = p_new;
            tmp = sigma_old;
            sigma_old = sigma_new;
            sigma_new = tmp;
            q_val_old = q_val_new;
        }
    }

    double *p_old_grad = (double *) malloc(n * n * sizeof(double));

    inv_p_old = 1 / p_old;

    for(int k = 0; k < n; k++){
        p_sigma_deriv(theta, n, sigma_old, mutation_num, k, p_old_grad + k*n);
    }

    for(int i = 0; i < m; i++){
        q_val_new = draw_from_q(theta, n, state_as_array, mutation_num, sigma_new);
        p_new = p_sigma(theta, n, sigma_new, mutation_num, mem_buffer);
        p_accept = (p_new * q_val_old) / (p_old * q_val_new);

        if (p_accept > 1){
            p_accept = 1;
        }

        if (rand() <= p_accept * RAND_MAX){
            p_old = p_new;
            inv_p_old = 1 / p_old;
            tmp = sigma_old;
            sigma_old = sigma_new;
            sigma_new = tmp;
            q_val_old = q_val_new;

            for(int k = 0; k < n; k++){
                p_sigma_deriv(theta, n, sigma_old, mutation_num, k, p_old_grad + k*n);
            }
        }

        score += p_new / q_val_new;

        for(int k = 0; k < n*n; k++){
            grad_out[k] += inv_p_old * p_old_grad[k];
        }
    }

    double inv_m = 1 / (1. * m);
    for(int k = 0; k < n*n; k++){
        grad_out[k] *= inv_m;
    }

    free(p_old_grad);
    free(state_as_array);
    free(mem_buffer);
    free(sigma_old);
    free(sigma_new);

    return log(score * inv_m);
}


/**
 * approximates the score and its gradient for a given MHN for a data set containing mutation data of tumors
 *
 * @param[in] theta array containing the theta entries of the MHN
 * @param[in] n total number of genes considered by the MHN, also column and row size of theta
 * @param[in] mutation_data array of States, each representing a tumor sample
 * @param[in] data_size number of tumor samples in the mutation data
 * @param[in] m number of paths sampled to compute the approximation
 * @param[in] burn_in_samples number of burn-in samples used before starting the approximation
 * @param[out] grad_out array which will contain the resulting gradient at the end
 *
 * @return score of the given MHN for the given mutation data
 */
double gradient_and_score_c(double *theta, int n, State *mutation_data, int data_size, int m, int burn_in_samples, double *grad_out){

    double *approx_grad = (double *) malloc(n * n * sizeof(double));
    double score = 0;

    for(int i = 0; i < data_size; i++){
        score += approximate_gradient_and_score(theta, n, &mutation_data[i], m, burn_in_samples, approx_grad);

        for(int j = 0; j < n*n; j++){
            grad_out[j] += approx_grad[j];
        }
    }

    double inv_data_size = 1 / (1. * data_size);
    for(int j = 0; j < n*n; j++){
        grad_out[j] *= inv_data_size;
    }

    free(approx_grad);
    return score / data_size;
}

#ifdef __cplusplus
}
#endif