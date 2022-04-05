# by Stefan Vocht
#
# in this script we implement the gradient approximation proposed by
# Gotovos et al. in "Scaling up Continuous-Time Markov Chains Helps Resolve Underspecification" (2021)
# it also contains a few functions to check the implementation for correctness

import numpy as np
import random


def q_next(theta: np.ndarray, curr_sequence: np.ndarray, new_element: int) -> float:
    """
    This function computes q_{sigma_[i-1] -> sigma_[i]} as used in eq. 6

    :param theta: theta matrix parametrizing the MHN
    :param curr_sequence: sigma_[i-1], sequence before adding the new element
    :param new_element: new element to be added to the given sequence
    :return:
    """
    theta_i = np.exp(theta[new_element, :])
    result = 1

    for element in curr_sequence:
        result *= theta_i[element]

    result *= theta_i[new_element]
    return result


def q_next_deriv(theta: np.ndarray, curr_sequence: np.ndarray, new_element: int, i: int) -> np.ndarray:
    """
    Compuates the derivative of q_next for all theta_i*

    :param theta:
    :param curr_sequence:
    :param new_element:
    :param i:
    :return:
    """
    result = np.zeros(theta.shape[0])

    if i != new_element:
        return result

    q_n = q_next(theta, curr_sequence, new_element)

    result[i] = q_n
    result[curr_sequence] = q_n

    return result


def q_tilde(theta: np.ndarray, sequence: np.ndarray) -> float:
    """
    This function computes the q with a tilde used in eq. 6, which represents a diagonal element of Q

    :param theta: theta matrix parametrizing the MHN
    :param sequence: sequence of mutated genes (sorted!)
    :return:
    """
    result = 0

    for i in range(theta.shape[0]):
        if i not in sequence:
            r_loc = 0
            for element in sequence:
                r_loc += theta[i, element]
            r_loc += theta[i, i]
            result += np.exp(r_loc)

    return result


def q_tilde_deriv(theta: np.ndarray, sequence: np.ndarray, i: int) -> np.ndarray:
    """
    Computes the derivate of q_tilde for all theta_i*

    :param theta:
    :param sequence:
    :param i:
    :return:
    """

    result = np.zeros(theta.shape[0])
    if i in sequence:
        return result

    q_n = q_next(theta, sequence, i)

    result[i] = q_n
    result[sequence] = q_n

    return result


def p_sigma(theta: np.ndarray, sequence: np.ndarray) -> float:
    """
    Computes the probability to observe the a sequence sigma according to eq. 6

    :param theta: theta matrix parametrizing the MHN
    :param sequence: sequence of mutated genes
    :return:
    """

    result = 1

    for i in range(len(sequence)):
        result *= q_next(theta, sequence[:i], sequence[i])
        result /= 1 + q_tilde(theta, sequence[:i])

    result /= 1 + q_tilde(theta, sequence)
    return result


def p_sigma_deriv(theta: np.ndarray, sequence: np.ndarray, k: int) -> np.ndarray:
    """
    Computes the derivative of p_sigma for all theta_k*

    :param theta:
    :param sequence:
    :param k:
    :return:
    """

    result = np.zeros(theta.shape[0])

    p_sig = p_sigma(theta, sequence)

    for i in range(len(sequence)):
        p_loc = p_sig
        one_plus_tilde = 1 + q_tilde(theta, sequence[:i])
        q_n = q_next(theta, sequence[:i], sequence[i])
        # squared in denominator of deriv, but partially compensated by removing the corresponding factor in p_sig
        p_loc /= one_plus_tilde
        p_loc /= q_n
        p_loc *= one_plus_tilde * q_next_deriv(theta, sequence[:i], sequence[i], k) - q_n * q_tilde_deriv(theta, sequence[:i], k)
        result += p_loc

    one_plus_tilde = 1 + q_tilde(theta, sequence)
    p_sig /= one_plus_tilde
    p_sig *= q_tilde_deriv(theta, sequence, k)

    result -= p_sig
    return result


def draw_from_q(theta: np.ndarray, s: np.ndarray) -> (np.ndarray, float):
    """
    Implementation of "Drawing from proposal Q" as shown in Appendix E of the paper

    :param theta: theta matrix parametrizing the MHN
    :param s: number array representing mutated genes
    :return: new sequence sigma and Q_val
    """

    q_val = 1  # in paper this is set to 0, but that makes no sense
    sigma = []

    for k in range(s.size):
        s_without_sigma = [v for v in s if v not in sigma]
        u = []
        for v in s_without_sigma:
            sigma_loc = sigma + [v]
            dv = 1
            for j in range(theta.shape[0]):
                if j in sigma_loc:
                    continue
                dv += np.exp(theta[j, j] + sum(theta[j, i] for i in sigma_loc))

            u.append(np.exp(sum(theta[j, v] for j in s_without_sigma) - theta[v, v]) / dv)

        sum_u = sum(u)
        x = np.random.choice(s_without_sigma, size=1, p=[u_elem / sum_u for u_elem in u])
        sigma.append(x[0])
        q_val *= u[s_without_sigma.index(x)] / sum_u

    return np.array(sigma, dtype=np.int), q_val


def approx_gradient(theta: np.ndarray, state: int, m: int = 50, burn_in_samples: int = 10) -> np.ndarray:
    """
    Implements the approximated gradient as shown in eq. 7

    :param theta:
    :param state: Integer where the 1s in binary form represent mutations
    :param m: number of samples taken for computing the gradient
    :param burn_in_samples: number of samples taken at the beginning without adding to the gradient
    :return: approximated gradient
    """

    n = theta.shape[0]
    state_as_array = np.array([i for i in range(n) if state >> i & 1])

    sigma_old, q_val_old = draw_from_q(theta, state_as_array)
    p_old = p_sigma(theta, sigma_old)

    for _ in range(burn_in_samples):
        sigma_new, q_val_new = draw_from_q(theta, state_as_array)
        p_new = p_sigma(theta, sigma_new)
        p_accept = min((1, (p_new * q_val_old) / (p_old * q_val_new)))

        if random.random() < p_accept:
            p_old = p_new
            sigma_old = sigma_new
            q_val_old = q_val_new

    resulting_gradient = np.zeros(theta.shape)
    p_old_grad = np.vstack([p_sigma_deriv(theta, sigma_old, k) for k in range(n)])

    for _ in range(m):
        sigma_new, q_val_new = draw_from_q(theta, state_as_array)
        p_new = p_sigma(theta, sigma_new)
        p_accept = min((1, (p_new * q_val_old) / (p_old * q_val_new)))

        if random.random() < p_accept:
            p_old = p_new
            sigma_old = sigma_new
            q_val_old = q_val_new
            p_old_grad = np.vstack([p_sigma_deriv(theta, sigma_old, k) for k in range(n)])

        resulting_gradient += (1 / p_old) * p_old_grad

    resulting_gradient /= m
    return resulting_gradient


def gradient(theta: np.ndarray, mutation_data: list, m: int = 50, burn_in_samples: int = 10) -> np.ndarray:
    """
    Computes the complete gradient for given mutation data

    :param theta: current theta matrix
    :param mutation_data: list containing integers for each sample, where the integer represent the mutations in the sample
    :param m: number of samples taken for each approxiamte gradient
    :param burn_in_samples: burn-in-samples used in approximate gradient
    :return: gradient
    """

    n = theta.shape[0]
    final_gradient = np.zeros((n, n))
    for state in mutation_data:
        final_gradient += approx_gradient(theta, state, m, burn_in_samples)

    return final_gradient / len(mutation_data)


def test_p_implementation(n: int = 6):
    """
    This function tests the implementation of eq. 6 and its derivative for each possible state for a given n

    :param n: number of genes that would be considered in a MHN
    """
    from MHN import Likelihood, ModelConstruction
    import StateSpaceRestriction
    import itertools

    np.random.seed(0)

    theta_ = ModelConstruction.random_theta(n)
    pth = Likelihood.generate_pTh(theta_)

    for state in range(1, 2 ** n):
        sigma0 = [i for i in range(n) if state >> i & 1]
        sigma_perms = itertools.permutations(sigma0)
        p_grad_sum = np.zeros((n, n))
        p_sum = 0
        for perm in sigma_perms:
            p = np.array(perm)
            p_sum += p_sigma(theta_, p)
            for k in range(n):
                p_grad_sum[k, :] += p_sigma_deriv(theta_, p, k)

        g = StateSpaceRestriction.restricted_gradient(theta_, state)
        p_grad_sum /= p_sum

        if abs(p_sum - pth[state]) > 1e-10 or np.any(abs(p_grad_sum - g) > 1e-10):
            print("Results dont match!")
            print("State: {:b}".format(state))
            print(f"{p_sum} vs. {pth[state]}")
            print("Gradients:")
            print(p_grad_sum)
            print(g)
            print(abs(p_grad_sum - g) < 1e-10)
            print("="*30)
        else:
            print("Correct result")


def test_drawing_q(n: int = 6, sequence: np.ndarray = None, iteration_num: int = 10_000):
    """
    Tests if the q returned by draw_from_q() actually is the correct ratio of the corresponding sequence

    :param n: total number of mutatable genes
    :param sequence: the sequence we want to draw permutations from using draw_from_q()
    :param iteration_num: number of iterations used to compute the true ratio
    :return:
    """

    from MHN import ModelConstruction
    from collections import defaultdict

    np.random.seed(0)
    random.seed(0)

    theta = ModelConstruction.random_theta(n)
    if sequence is None:
        sequence = np.arange(n)

    d = defaultdict(int)
    q_values = dict()

    for _ in range(iteration_num):
        sigma, q = draw_from_q(theta, sequence)
        hashable = tuple(sigma.tolist())
        d[hashable] += 1
        q_values[hashable] = q

    for sigma, q in q_values.items():
        print(d[sigma] / iteration_num, q)


def test_sampling():
    """
    This function tests if the Cython implementation for "Algorithm 1" in the Gotovos paper
    that draws random samples according to the distribution yielded by a given MHN is correct
    """

    n = 5

    import matplotlib.pyplot as plt
    import approximate_gradient_cython as agc
    from MHN import ModelConstruction, Likelihood

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    rect_scatter = [left, bottom, width, height]
    fig = plt.figure()
    ax = fig.add_axes(rect_scatter)
    theta = ModelConstruction.random_theta(n)

    cython_sample = [agc.draw_one_sample(np.exp(theta)) for _ in range(100_000)]
    actual_sample = np.random.choice(range(2**n), 100_000, p=Likelihood.generate_pTh(theta))

    ax.hist([actual_sample, cython_sample], density=True, bins=2**n)

    plt.show()


if __name__ == '__main__':
    # test_p_implementation()
    # test_drawing_q(sequence=np.array([0, 2, 3, 5]))
    test_sampling()
