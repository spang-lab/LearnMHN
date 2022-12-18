# by Stefan Vocht
#
# this script contains classes that can be used to optimize a MHN for given data
#
import warnings
from enum import Enum

from .ssr.learn_MHN import learn_MHN, build_regularized_score_func, build_regularized_gradient_func
from .ssr.state_storage import StateStorage
from .ssr.state_space_restriction import CUDAError, cuda_available, CUDA_AVAILABLE
from .ssr.state_space_restriction import gradient_and_score, cython_gradient_and_score

if cuda_available() == CUDA_AVAILABLE:
    from .ssr.state_space_restriction import cuda_gradient_and_score
else:
    cuda_gradient_and_score = None

import numpy as np
from numpy import genfromtxt


class Device(Enum):
    """
    A small Enum which can represent device types
    """
    AUTO, CPU, GPU = range(3)


class StateSpaceOptimizer:
    """
    This optimizer uses state space restriction to optimize an MHN
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

        self.__gradient_and_score_func = gradient_and_score

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

    def set_device(self, device: Device):
        """
        Set the device that should be used for training. You have three options:

        Device.AUTO: (default) automatically select the device that best fits the data
        Device.CPU:  use the CPU implementations to compute the scores and gradients
        Device.GPU:  use the GPU/CUDA implementations to compute the scores and gradients
        """
        if not isinstance(device, Device):
            raise ValueError(f"The given device is not an instance of {Device}")
        if device == Device.GPU and cuda_gradient_and_score is None:
            raise CUDAError(cuda_available())
        self.__gradient_and_score_func = {
            Device.AUTO: gradient_and_score,
            Device.CPU: cython_gradient_and_score,
            Device.GPU: cuda_gradient_and_score
        }[device]

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

        score_func = build_regularized_score_func(self.__gradient_and_score_func)
        gradient_func = build_regularized_gradient_func(self.__gradient_and_score_func)

        self.__result = learn_MHN(self.__data, self.__init, lam, maxit, trace, reltol,
                                  round_result, callback_func, score_func, gradient_func)

        self.__backup_current_step = None
        return self.__result

    @property
    def result(self):
        return self.__result

    @property
    def bin_datamatrix(self):
        return self.__bin_datamatrix

