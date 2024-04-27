"""
This submodule contains functions to learn an MHN.
"""
# author(s): Stefan Vocht, Y. Linda Hu

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from typing import Callable

from .state_containers import StateContainer, create_indep_model


def learn_mhn(states: StateContainer, score_func: Callable, jacobi: Callable, init: np.ndarray = None, lam: float = 0,
              maxit: int = 5000, trace: bool = False, reltol: float = 1e-07, round_result: bool = True,
              callback: Callable = None) -> OptimizeResult:
    """
    This function is used to train an MHN.

    :param states: a StateContainer object containing all mutation states observed in the data
    :param init: starting point for the training (initial theta)
    :param lam: tuning parameter lambda for regularization
    :param maxit: maximum number of training iterations
    :param trace: set to True to print convergence messages (see scipy.optimize.minimize)
    :param reltol: Gradient norm must be less than reltol before successful termination (see "gtol" scipy.optimize.minimize)
    :param round_result: if True, the result is rounded to two decimal places
    :param callback: function called after each iteration, must take theta as argument
    :param score_func: score function used for training
    :param jacobi: gradient function used for training
    :return: OptimizeResult object containing the trained model
    """

    n = states.get_data_shape()[1]

    if init is None:
        init = create_indep_model(states)

    init_shape = init.shape
    init = init.flatten()

    # this container is given to the score and gradient function to communicate with each other
    score_and_gradient_container = [None, None]

    opt = minimize(fun=score_func, x0=init, args=(states, lam, n, score_and_gradient_container), method="L-BFGS-B",
                   jac=jacobi, options={'maxiter': maxit, 'disp': trace, 'gtol': reltol}, callback=callback)

    opt.x = opt.x.reshape(init_shape)

    if round_result:
        opt.x = np.around(opt.x, decimals=2)

    return opt
