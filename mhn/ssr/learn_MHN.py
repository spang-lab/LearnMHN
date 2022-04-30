# by Stefan Vocht
#
# this script is used to learn a MHN using state space restriction or the approximated gradient
#

import numpy as np
from scipy.optimize import minimize

from typing import Callable

from state_storage import State_storage, create_indep_model
import StateSpaceRestrictionCython
import approximate_gradient_cython as agc


def L1(theta: np.ndarray, eps: float = 1e-05) -> float:
    """
    Computes the L1 penalty
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return np.sum(np.sqrt(theta_**2 + eps))


def L1_(theta: np.ndarray, eps: float = 1e-05) -> np.ndarray:
    """
    Derivative of the L1 penalty
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return theta_ / np.sqrt(theta_**2 + eps)


def reg_state_space_restriction_score(theta: np.ndarray, states: State_storage, lam: float,
                                      n: int, score_grad_container: list) -> float:
    """
    Computes the score using state space restriction with L1 regularization

    :param theta: current theta
    :param states: states observed in the data
    :param lam: regularization parameter
    :param n: size of theta (nxn)
    :param score_grad_container: a list that enables this function to communicate with the gradient function
    :return: regularized score
    """
    theta = theta.reshape((n, n))

    # grad, score = StateSpaceRestrictionCython.gradient_and_score(theta, states)
    print("Start computing gradient")
    grad, score = StateSpaceRestrictionCython.gradient_and_score_with_cuda(theta, states)
    print("Finish")
    score_grad_container[0] = grad

    return -(score - lam * L1(theta))


def reg_state_space_restriction_gradient(theta: np.ndarray, states: State_storage, lam: float,
                                         n: int, score_grad_container: list) -> np.ndarray:
    """
    Computes the gradient state space restriction with L1 regularization

    :param theta: current theta
    :param states: states observed i the data
    :param lam: regularization parameter
    :param n: size of theta (nxn)
    :param score_grad_container: a list that enables this function to communicate with the score function
    :return: regularized gradient
    """

    n = n or int(np.sqrt(theta.size))
    theta_ = theta.reshape((n, n))

    grad = score_grad_container[0]
    if grad is None:
        grad, score = StateSpaceRestrictionCython.gradient(theta, states)

    return -(grad - lam * L1_(theta_)).flatten()


def reg_approximate_score(theta: np.ndarray, states: State_storage, lam: float,
                                      n: int, score_grad_container: list) -> float:
    """
    Computes the score using the approximate score with L1 regularization

    :param theta: current theta
    :param states: states observed in the data
    :param lam: regularization parameter
    :param n: size of theta (nxn)
    :param score_grad_container: a list that enables this function to communicate with the gradient function
    :return: regularized score
    """
    theta = theta.reshape((n, n))

    # grad, score = StateSpaceRestrictionCython.gradient_and_score(theta, states)
    print("Start approximating gradient")
    grad, score = agc.gradient_and_score_using_c(np.exp(theta), states, 50, 10)
    print("Finish")
    score_grad_container[0] = grad

    return -(score - lam * L1(theta))


def reg_approximate_gradient(theta: np.ndarray, states: State_storage, lam: float,
                                         n: int, score_grad_container: list) -> np.ndarray:
    """
    Computes the gradient state space restriction with L1 regularization

    :param theta: current theta
    :param states: states observed i the data
    :param lam: regularization parameter
    :param n: size of theta (nxn)
    :param score_grad_container: a list that enables this function to communicate with the score function
    :return: regularized gradient
    """

    n = n or int(np.sqrt(theta.size))
    theta_ = theta.reshape((n, n))

    grad = score_grad_container[0]
    if grad is None:
        grad, score = agc.gradient_and_score_using_c(np.exp(theta), states, 50, 10)

    return -(grad - lam * L1_(theta_)).flatten()


def learn_MHN(states: State_storage, init: np.ndarray = None, lam: float = 0, maxit: int = 5000,
              trace: bool = False, reltol: float = 1e-07, round_result: bool = True, callback: Callable = None,
              score_func: Callable = reg_state_space_restriction_score,
              jacobi: Callable = reg_state_space_restriction_gradient) -> np.ndarray:
    """
    This function is used to train a MHN, it is very similar to the learn_MHN function from the original MHN

    :param states: a State_storage object containing all mutation states observed in the data
    :param init: starting point for the training (initial theta)
    :param lam: tuning parameter for regularization
    :param maxit: maximum number of training iterations
    :param trace: set to True to print convergence messages (see scipy.optimize.minimize)
    :param reltol: Gradient norm must be less than reltol before successful termination (see "gtol" scipy.optimize.minimize)
    :param round_result: if True, the result is rounded to two decimal places
    :param callback: function called after each iteration, must take theta as argument
    :param score_func: score function used for training
    :param jacobi: gradient function used for training
    :return: trained model
    """

    n = states.get_data_shape()[1]

    if init is None:
        init = create_indep_model(states)

    # this container is given to the score and gradient function to communicate with each other
    score_and_gradient_container = [None, None]

    opt = minimize(fun=score_func, x0=init, args=(states, lam, n, score_and_gradient_container), method="L-BFGS-B",
                   jac=jacobi, options={'maxiter': maxit, 'disp': trace, 'gtol': reltol}, callback=callback)

    theta = opt.x.reshape((n, n))

    if round_result:
        theta = np.around(theta, decimals=2)

    return theta


def learn_small_gbm():
    """
    This is a test function that learns a MHN on the Glioblastoma data used in the original paper
    """
    from numpy import genfromtxt
    my_data = genfromtxt('../data/small_gbm.csv', delimiter=',', dtype=np.int)

    bin_data = my_data[1:, 1:]

    states = State_storage(bin_data)
    print(bin_data)
    print(states.get_data_shape())

    print("Start")
    learned_theta = learn_MHN(states, lam=0.01)

    print(learned_theta)

    print(np.exp(learned_theta))

    with open('../data/small_gbm_mhn.npy', 'wb') as f:
        np.save(f, np.exp(learned_theta))


def heatmap_for_small_gbm():
    """
    This function is used to plot the final test MHN as a heatmap
    """
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    with open('../data/small_gbm_mhn.npy', 'rb') as f:
        data = np.load(f)

    genes = ['CDKN2A(D)', 'TP53(M)', 'PTEN(M)', 'NF1(M)', 'CDK4(A)', 'EGFR(M)', 'PDGFRA(A)', 'MDM4(A)', 'PTEN(D)',
             'RB1(D)', 'FAF1(D)', 'SPTA1(M)', 'MDM2(A)', 'TP53(D)', 'PAOX(M)', 'OBSCN(M)', 'LRP2(M)', 'PIK3CA(M)',
             'CNTNAP2(M)', 'IDH1(M)']

    df = pd.DataFrame(np.log(data), columns=genes)

    p1 = sns.heatmap(df, xticklabels=genes, yticklabels=genes, cmap='rainbow')
    plt.show()


def learn_new_mhn(np_name: str, csv_name: str, method: str = "SSR", save_progress_file: str = None):
    """
    Use this function to learn a new MHN

    :param np_name: file that will contain the trained MHN as a numpy matrix
    :param csv_name: csv file containing the training data
    :param method: method to compute the score and gradient,
    either "SSR" for State Space Restriction or "AG" for approximated gradient
    :param save_progress_file: if given (recommended), every 10th step will be stored in this file as backup
    """
    from numpy import genfromtxt
    my_data = genfromtxt(csv_name, delimiter=';', dtype=np.int)

    bin_data = my_data[1:, 1:]

    print(bin_data.shape)
    states = State_storage(bin_data)
    print(bin_data)

    print(states.get_data_shape())

    if not save_progress_file:
        print("Warning: No file to save progress was given, no backups will be made during training!")

    def callback_function(theta):
        print(callback_function.counter)
        callback_function.counter += 1

        if save_progress_file and not (callback_function.counter % 10):
            with open(save_progress_file, 'wb') as f:
                np.save(f, theta)

    callback_function.counter = 0

    if method == "SSR":
        score_func = reg_state_space_restriction_score
        grad_func = reg_state_space_restriction_gradient
    else:
        score_func = reg_approximate_score
        grad_func = reg_approximate_gradient

    print("Start")
    learned_theta = learn_MHN(states, lam=0.01, callback=callback_function,
                              score_func=score_func,
                              jacobi=grad_func)

    with open(np_name, 'wb') as f:
        np.save(f, np.exp(learned_theta))


def heatmap_of_mhn(np_name: str, csv_name: str):
    """
    Create a heatmap for a given MHN

    :param np_name: file that contains the MHN as a numpy matrix
    :param csv_name: csv file containing the data (and especially the gene names!) used to train the MHN
    """
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    import csv

    with open(np_name, 'rb') as f:
        data = np.load(f)

    with open(csv_name) as f:
        genes = next(csv.reader(f, delimiter=';'))[1:]

    df = pd.DataFrame(np.log(data))

    p1 = sns.heatmap(df, xticklabels=genes, yticklabels=genes, cmap='seismic')
    plt.show()


def main():
    learn_new_mhn('../data/test_mhn.npy', '../data/gbm_middle.csv', save_progress_file='../data/tmp_mhn.npy')
    heatmap_of_mhn('../data/test_mhn.npy', '../data/gbm_middle.csv')


if __name__ == '__main__':
    main()

