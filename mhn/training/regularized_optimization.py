"""
This submodule contains functions to learn an MHN.
"""
# author(s): Stefan Vocht, Y. Linda Hu

from typing import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .state_containers import StateContainer, create_indep_model


def learn_mhn(states: StateContainer, score_func: Callable, jacobi: Callable, init: np.ndarray = None, lam: float = 0,
              maxit: int = 5000, trace: bool = False, reltol: float = 1e-07, round_result: bool = True,
              callback: Callable = None) -> OptimizeResult:
    """
    Trains an MHN.

    Args:
        states (StateContainer): A container object holding all mutation states observed in the data.
        score_func (Callable): The score function used for training.
        jacobi (Callable): The gradient function used for training.
        init (np.ndarray, optional): Initial theta for training. If None is given, an independence model is used. Defaults to None.
        lam (float, optional): Regularization tuning parameter lambda. Defaults to 0.
        maxit (int, optional): Maximum number of training iterations. Defaults to 5000.
        trace (bool, optional): If True, prints convergence messages (see `scipy.optimize.minimize`). Defaults to False.
        reltol (float, optional): Gradient norm threshold for successful termination (see "gtol" in `scipy.optimize.minimize`). Defaults to 1e-07.
        round_result (bool, optional): If True, rounds the result to two decimal places. Defaults to True.
        callback (Callable, optional): A function called after each iteration, taking theta as an argument. Defaults to None.

    Returns:
        OptimizeResult: The result of the optimization containing the trained model.
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
