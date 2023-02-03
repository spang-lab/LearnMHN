# by Stefan Vocht
#
# this script contains classes that can be used to optimize a MHN for given data
#
import warnings
from enum import Enum
from abc import ABC, abstractmethod

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


class _Optimizer(ABC):
    """
    This abstract Optimizer class is the base class for the other Optimizer classes and cannot be instantiated alone
    """
    def __init__(self):
        self._data = None
        self._bin_datamatrix = None
        self.__result = None
        self.__init_theta = None
        self.__custom_callback = None

        self.__backup_steps = -1
        self.__backup_filename = None
        self.__backup_always_new_file = False
        self.__backup_current_step = None

        self._gradient_and_score_func = None

    def set_init_theta(self, init: np.ndarray):
        """
        Use this method to set a theta as starting point for learning a new MHN.
        If none is given, the optimization starts with an independence model.
        """
        self.__init_theta = init
        return self

    def set_callback_func(self, callback=None):
        """
        Use this method to set a callback function called after each iteration in the BFGS algorithm
        """
        if callback is not None and not hasattr(callback, '__call__'):
            raise ValueError("callback has to be a function!")
        self.__custom_callback = callback
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

        :param lam: tuning parameter for the L1 regularization
        :param maxit: maximum number of training iterations
        :param trace: set to True to print convergence messages (see scipy.optimize.minimize)
        :param reltol: Gradient norm must be less than reltol before successful termination (see "gtol" scipy.optimize.minimize)
        :param round_result: if True, the result is rounded to two decimal places
        :return: trained model
        """
        if self._data is None:
            raise ValueError("You have to load data before training!")

        self.__result = None
        self.__backup_current_step = 0

        if self.__custom_callback is None and self.__backup_steps < 1:
            callback_func = None
        else:
            callback_func = self.__total_callback_func

        score_func = build_regularized_score_func(self._gradient_and_score_func)
        gradient_func = build_regularized_gradient_func(self._gradient_and_score_func)

        self.__result = learn_MHN(self._data, self.__init_theta, lam, maxit, trace, reltol,
                                  round_result, callback_func, score_func, gradient_func)

        self.__backup_current_step = None
        return self.__result

    @property
    def result(self) -> np.ndarray:
        """
        The resulting Theta matrix after training, same as the return value of the train() method.
        This property mainly exists as a kind of backup to ensure that the result of the training is not lost, if the
        user forgets to save the returned value of the train() method in a variable.
        """
        return self.__result

    @property
    def bin_datamatrix(self) -> np.ndarray:
        """
        The mutation matrix used as training data to learn an MHN
        """
        return self._bin_datamatrix

    @staticmethod
    def _preprocess_binary_matrix(data_matrix: np.ndarray) -> np.ndarray:
        """
        This function is used to make sure that the given data matrix is in the correct format.
        Correct format:
            a 2D numpy array with dtype=np.int32, which only contains 0s and 1s

        While the dtype will be changed automatically, if not np.int32, a matrix not being 2D or containing other values
        than 0s and 1s will raise a ValueError.

        :return: the given data_matrix with its dtype set to np.int32
        """
        if len(data_matrix.shape) != 2:
            raise ValueError("The given data matrix must be two-dimensional")
        # StateStorage only accepts numpy arrays with dtype=np.int32
        if data_matrix.dtype != np.int32:
            data_matrix = data_matrix.astype(dtype=np.int32)
            warnings.warn("The dtype of the given data matrix is changed to np.int32")
        if not set(data_matrix.flatten()).issubset({0, 1}):
            raise ValueError("The data matrix must only contain 0s and 1s")

        return data_matrix

    @abstractmethod
    def set_device(self, device: "_Optimizer.Device"):
        """
        Set the device that should be used for training. You have three options:

        Device.AUTO: (default) automatically select the device that best fits the data
        Device.CPU:  use the CPU implementations to compute the scores and gradients
        Device.GPU:  use the GPU/CUDA implementations to compute the scores and gradients
        """
        pass

    class Device(Enum):
        """
        A small Enum which can represent device types
        """
        AUTO, CPU, GPU = range(3)


class StateSpaceOptimizer(_Optimizer):
    """
    This optimizer uses state space restriction to optimize an MHN
    """
    def __init__(self):
        super().__init__()
        self._gradient_and_score_func = gradient_and_score

    def load_data_matrix(self, data_matrix: np.ndarray):
        """
        Load binary mutation data stored in a numpy array

        :param data_matrix: two-dimensional numpy array which should have dtype=np.int32
        :return: this optimizer object
        """
        data_matrix = self._preprocess_binary_matrix(data_matrix)
        self._data = StateStorage(data_matrix)
        self._bin_datamatrix = data_matrix
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

    def set_device(self, device: _Optimizer.Device):
        """
        Set the device that should be used for training. You have three options:

        Device.AUTO: (default) automatically select the device that best fits the data
        Device.CPU:  use the CPU implementations to compute the scores and gradients
        Device.GPU:  use the GPU/CUDA implementations to compute the scores and gradients
        """
        if not isinstance(device, _Optimizer.Device):
            raise ValueError(f"The given device is not an instance of {_Optimizer.Device}")
        if device == _Optimizer.Device.GPU and cuda_gradient_and_score is None:
            raise CUDAError(cuda_available())
        self._gradient_and_score_func = {
            _Optimizer.Device.AUTO: gradient_and_score,
            _Optimizer.Device.CPU: cython_gradient_and_score,
            _Optimizer.Device.GPU: cuda_gradient_and_score
        }[device]

