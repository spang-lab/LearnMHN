# by Stefan Vocht
#
# this script contains classes that can be used to optimize a MHN for given data
#
import warnings

from .ssr.learn_MHN import learn_MHN, reg_state_space_restriction_score, reg_state_space_restriction_gradient
from .ssr.state_storage import StateStorage

import numpy as np
from numpy import genfromtxt


class StateSpaceOptimizer:
    """
    This optimizer uses state space restriction to optimize a MHN
    """
    def __init__(self):
        self.__data = None
        self.__bin_datamatrix = None
        self.__result = None
        self.__init = None
        self.__custom_callback = None

        self.__backup_steps = -1
        self.__backup_filename = None
        self.__backup_always_new_file = False
        self.__backup_current_step = None

        self.__score_func = reg_state_space_restriction_score
        self.__grad_func = reg_state_space_restriction_gradient

    def load_data_matrix(self, data_matrix: np.ndarray):
        """
        Load binary mutation data stored in a numpy array

        :param data_matrix: two-dimensional numpy array which should have dtype=np.int32
        :return: this optimizer object
        """
        if len(data_matrix.shape) != 2:
            raise ValueError("The given data matrix must be two-dimensional")
        # StateStorage only accepts numpy arrays with dtype=np.int32
        if data_matrix.dtype != np.int32:
            data_matrix = data_matrix.astype(dtype=np.int32)
            warnings.warn("The dtype of the given data matrix is changed to np.int32")
        if not set(data_matrix.flatten()).issubset({0, 1}):
            raise ValueError("The data matrix must only contain 0s and 1s")
        self.__data = StateStorage(data_matrix)
        self.__bin_datamatrix = data_matrix
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

    def set_callback_func(self, callback=None):
        if callback is not None and not hasattr(callback, '__call__'):
            raise ValueError("callback has to be a function!")
        self.__custom_callback = callback
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

    def save_progress(self, steps: int = -1, always_new_file: bool = False, filename: str = 'theta_backup.npy'):
        """
        If you want to regularly save the progress during training, you can use this function and define the number
        of steps between each progress save

        :param filename: file name of the backup file
        :param steps: number of training iterations between each progress backup
        :param always_new_file: if this is True, every backup is stored in a separate file, else the file is overwritten each time
        :return: this optimizer object
        """
        self.__backup_steps = steps
        self.__backup_always_new_file = always_new_file
        self.__backup_filename = filename
        return self

    def __total_callback_func(self, theta: np.ndarray):
        if self.__custom_callback is not None:
            self.__custom_callback(theta)

        if self.__backup_steps > 0:
            self.__create_backup(theta)

    def __create_backup(self, theta: np.ndarray):
        self.__backup_current_step += 1
        if (self.__backup_current_step % self.__backup_steps) == 0:
            filename = self.__backup_filename
            if self.__backup_always_new_file:
                try:
                    idx = filename.index(".")
                    filename = filename[:idx] + f"_{self.__backup_current_step}" + filename[idx:]
                except ValueError:  # str.index raises ValueError if no "." is present in the filename
                    filename += f"_{self.__backup_current_step}.npy"
            with open(filename, 'wb') as f:
                np.save(f, theta)

    def train(self, lam: float = 0, maxit: int = 5000, trace: bool = False,
              reltol: float = 1e-7, round_result: bool = True) -> np.ndarray:
        """
        Use this function to learn a new MHN from the data given to this optimizer.

        :param lam: tuning parameter for regularization
        :param maxit: maximum number of training iterations
        :param trace: set to True to print convergence messages (see scipy.optimize.minimize)
        :param reltol: Gradient norm must be less than reltol before successful termination (see "gtol" scipy.optimize.minimize)
        :param round_result: if True, the result is rounded to two decimal places
        :return: trained model
        """
        if self.__data is None:
            raise ValueError("You have to load data before training!")

        self.__result = None
        self.__backup_current_step = 0

        if self.__custom_callback is None and self.__backup_steps < 1:
            callback_func = None
        else:
            callback_func = self.__total_callback_func

        self.__result = learn_MHN(self.__data, self.__init, lam, maxit, trace, reltol,
                                  round_result, callback_func, self.__score_func, self.__grad_func)

        self.__backup_current_step = None
        return self.__result

    @property
    def result(self):
        return self.__result

    @property
    def bin_datamatrix(self):
        return self.__bin_datamatrix

