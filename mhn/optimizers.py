# by Stefan Vocht
#
# this script contains classes that can be used to optimize a MHN for given data
#


from .ssr.learn_MHN import learn_MHN, reg_state_space_restriction_score, reg_state_space_restriction_gradient
from .ssr.learn_MHN import reg_approximate_score, reg_approximate_gradient

from .ssr.state_storage import State_storage

import numpy as np
from numpy import genfromtxt


class StateSpaceOptimizer:
    """
    This optimizer uses state space restriction to optimize a MHN
    """
    def __init__(self):
        self.__data = None
        self.__result = None
        self.__init = None
        self.__callback = None

        self.__score_func = reg_state_space_restriction_score
        self.__grad_func = reg_state_space_restriction_gradient

    def load_data_matrix(self, data_matrix: np.ndarray):
        self.__data = State_storage(data_matrix)
        return self

    def load_data_from_csv(self, src: str, delimiter: str = ';',
                           first_row: int = None, last_row: int = None, first_col: int = None, last_col: int = None):
        """
        Load binary mutation data from a CSV file

        :param src: path to the CSV file
        :param delimiter:  delimiter used in the CSV file (default: ';')
        :param first_row: (Optional) first row of the CSV file that is part of the binary matrix without the column names
        :param last_row: (Optional) last row of the CSV file that is part of the binary matrix without the column names
        :param first_col: (Optional) first column of the CSV file that is part of the binary matrix without the row names
        :param last_col: (Optional) last column of the CSV file that is part of the binary matrix without the row names
        :return: this optimizer object
        """
        data_matrix = genfromtxt(src, delimiter=delimiter, dtype=np.int32)
        data_matrix = data_matrix[first_row: last_row, first_col: last_col]
        self.load_data_matrix(data_matrix)
        return self

    def set_init_theta(self, init: np.ndarray):
        self.__init = init
        return self

    def set_callback_func(self, callback):
        if not hasattr(callback, '__call__'):
            raise ValueError("callback has to be a function!")
        self.__callback = callback
        return self

    def set_score_and_gradient_function(self, score_func, gradient_func):
        if not hasattr(score_func, '__call__') or not hasattr(gradient_func, '__call__'):
            raise ValueError("score_func and gradient_func have to be functions!")
        self.__score_func = score_func
        self.__grad_func = gradient_func
        return self

    def use_state_space_restriction(self):
        self.__score_func = reg_state_space_restriction_score
        self.__grad_func = reg_state_space_restriction_gradient
        return self

    def use_approximate_gradient(self):
        self.__score_func = reg_approximate_score
        self.__grad_func = reg_approximate_gradient
        return self

    def train(self, lam: float = 0, maxit: int = 5000, trace: bool = False,
              reltol: float = 1e-7, round_result: bool = True) -> np.ndarray:
        if self.__data is None:
            raise ValueError("You have to load data before training!")

        self.__result = None
        self.__result = learn_MHN(self.__data, self.__init, lam, maxit, trace, reltol,
                                  round_result, self.__callback, self.__score_func, self.__grad_func)

        return self.__result

    @property
    def result(self):
        return self.__result

