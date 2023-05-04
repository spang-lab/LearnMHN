"""
This submodule contains Optimizer classes to learn an MHN from mutation data.
"""
# author(s): Stefan Vocht, Y. Linda Hu

from __future__ import annotations

import warnings
from enum import Enum
import abc

import numpy as np
import pandas as pd

from . import model

from mhn.ssr import regularized_optimization as reg_optim
from .ssr.state_containers import StateContainer, StateAgeContainer
from .ssr.state_space_restriction import CUDAError, cuda_available, CUDA_AVAILABLE
from mhn.ssr import state_space_restriction as marginalized_funcs

from mhn.ssr import matrix_exponential as mat_exp


class _Optimizer(abc.ABC):
    """
    This abstract Optimizer class is the base class for the other Optimizer classes and cannot be instantiated alone.
    """

    def __init__(self):
        self._data = None
        self._bin_datamatrix = None
        self.__result = None
        self._events = None

        self.__init_theta = None
        self.__custom_callback = None

        self.__backup_steps = -1
        self.__backup_filename = None
        self.__backup_always_new_file = False
        self.__backup_current_step = None

        self._gradient_and_score_func = None

    def set_init_theta(self, init: np.ndarray):
        """
        Use this method to set a theta as starting point for learning a new MHN. The theta must be in logarithmic form.

        If none is given, the optimization starts with an independence model, where the baseline hazard Theta_ii
        of each event is set to its empirical odds and the hazard ratios (off-diagonal entries) are set to exactly 1.
        """
        self.__init_theta = init
        return self

    def get_data_properties(self):
        """
        You can use this method to get some information about the loaded mutation data, e.g. how many events and samples
        are present in the data, how many mutations a sample has on average etc.

        :returns: a dictionary containing information about the data
        """
        if self._bin_datamatrix is None:
            return {}

        total_mutations_per_sample = np.sum(self._bin_datamatrix, axis=1)
        return {
            'samples': self._bin_datamatrix.shape[0],
            'events': self._bin_datamatrix.shape[1],
            'mutations per sample': {
                'mean': np.mean(total_mutations_per_sample),
                'median': np.median(total_mutations_per_sample),
                'max': np.max(total_mutations_per_sample),
                'min': np.min(total_mutations_per_sample)
            }
        }

    def set_callback_func(self, callback=None):
        """
        Use this method to set a callback function called after each iteration in the BFGS algorithm.
        The given function must take exactly one argument, namely the theta matrix computed in the last iteration.
        """
        if callback is not None and not hasattr(callback, '__call__'):
            raise ValueError("callback has to be a function!")
        self.__custom_callback = callback
        return self

    def save_progress(self, steps: int = -1, always_new_file: bool = False, filename: str = 'theta_backup.npy'):
        """
        If you want to regularly save the progress during training, you can use this function and define the number
        of steps between each progress save.

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
                    filename = filename[:idx] + \
                        f"_{self.__backup_current_step}" + filename[idx:]
                except ValueError:  # str.index raises ValueError if no "." is present in the filename
                    filename += f"_{self.__backup_current_step}.npy"
            with open(filename, 'wb') as f:
                np.save(f, theta)

    def train(self, lam: float = None, maxit: int = 5000, trace: bool = False,
              reltol: float = 1e-7, round_result: bool = True) -> model.MHN:
        """
        Use this function to learn a new MHN from the data given to this optimizer.

        :param lam: tuning parameter lambda for regularization (default: 1/(number of samples in the dataset))
        :param maxit: maximum number of training iterations
        :param trace: set to True to print convergence messages (see scipy.optimize.minimize)
        :param reltol: Gradient norm must be less than reltol before successful termination (see "gtol" scipy.optimize.minimize)
        :param round_result: if True, the result is rounded to two decimal places
        :return: trained model
        """
        if self._data is None:
            raise ValueError("You have to load data before training!")

        if lam is None:
            lam = 1 / self._data.get_data_shape()[0]

        self.__result = None
        self.__backup_current_step = 0

        if self.__custom_callback is None and self.__backup_steps < 1:
            callback_func = None
        else:
            callback_func = self.__total_callback_func

        score_func = reg_optim.build_regularized_score_func(self._gradient_and_score_func)
        gradient_func = reg_optim.build_regularized_gradient_func(self._gradient_and_score_func)

        result = reg_optim.learn_MHN(self._data, self.__init_theta, lam, maxit, trace, reltol,
                                     round_result, callback_func, score_func, gradient_func)

        self.__backup_current_step = None

        self.__result = model.MHN(
            log_theta=result.x,
            events=self._events,
            meta={
                "lambda": lam,
                "init": self.__init_theta,
                "maxit": maxit,
                "reltol": reltol,
                "score": result.fun,
                "message": result.message,
                "status": result.status,
                "nit": result.nit
            })

        return self.__result

    @property
    def result(self) -> model.MHN:
        """
        The resulting MHN after training, same as the return value of the train() method.
        This property mainly exists as a kind of backup to ensure that the result of the training is not lost, if the
        user forgets to save the returned value of the train() method in a variable.
        """
        return self.__result

    @property
    @abc.abstractmethod
    def training_data(self):
        """
        This property returns all the data given to this optimizer to train a new MHN.
        """
        pass

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
        # StateContainer only accepts numpy arrays with dtype=np.int32
        if data_matrix.dtype != np.int32:
            data_matrix = data_matrix.astype(dtype=np.int32)
            warnings.warn("The dtype of the given data matrix is changed to np.int32")
        if not set(data_matrix.flatten()).issubset({0, 1}):
            raise ValueError("The data matrix must only contain 0s and 1s")

        return data_matrix

    @abc.abstractmethod
    def set_device(self, device: "_Optimizer.Device"):
        """
        Set the device that should be used for training.

        You have three options:
            Device.AUTO: (default) automatically select the device that is likely to match the data
            Device.CPU:  use the CPU implementations to compute the scores and gradients
            Device.GPU:  use the GPU/CUDA implementations to compute the scores and gradients

        The Device enum is part of this optimizer class.
        """
        if not isinstance(device, _Optimizer.Device):
            raise ValueError(f"The given device is not an instance of {_Optimizer.Device}")

        return self

    class Device(Enum):
        """
        A small Enum which can represent device types.
        """
        AUTO, CPU, GPU = range(3)


class StateSpaceOptimizer(_Optimizer):
    """
    This optimizer uses state-space restriction to optimize an MHN on data with no age information for the individual
    samples.
    """
    def __init__(self):
        super().__init__()
        self._gradient_and_score_func = marginalized_funcs.gradient_and_score

    def load_data_matrix(self, data_matrix: np.ndarray | pd.DataFrame):
        """
        Load mutation data stored in a numpy array or pandas DataFrame, where the rows represent samples and
        columns represent genes.
        Mutations of genes are represented by 1s, intact genes are represented by 0s.

        :param data_matrix: either a pandas DataFrame or a two-dimensional numpy array which should have dtype=np.int32
        :return: this optimizer object
        """
        if isinstance(data_matrix, pd.DataFrame):
            self._events = data_matrix.columns.to_list()
            data_matrix = np.array(data_matrix, dtype=np.int32)
        data_matrix = self._preprocess_binary_matrix(data_matrix)
        self._data = StateContainer(data_matrix)
        self._bin_datamatrix = data_matrix
        return self

    def load_data_from_csv(self, src: str, delimiter: str = ',', **kwargs):
        """
        Load mutation data from a CSV file. The rows have to represent samples and the columns represent genes.
        Mutations of genes are represented by 1s, intact genes are represented by 0s.

        :param src: path to the CSV file
        :param delimiter:  delimiter used in the CSV file (default: ',')
        :param kwargs: all additional keyword arguments are passed on to pandas' read_csv() function
        :return: this optimizer object
        """
        df = pd.read_csv(src, delimiter=delimiter, **kwargs)
        self.load_data_matrix(df)
        return self

    def set_device(self, device: "StateSpaceOptimizer.Device"):
        """
        Set the device that should be used for training.

        You have three options:
            Device.AUTO: (default) automatically select the device that is likely to match the data
            Device.CPU:  use the CPU implementations to compute the scores and gradients
            Device.GPU:  use the GPU/CUDA implementations to compute the scores and gradients

        The Device enum is part of this optimizer class.
        """
        super().set_device(device)
        if device == _Optimizer.Device.GPU:
            if cuda_available() != CUDA_AVAILABLE:
                raise CUDAError(cuda_available())

            self._gradient_and_score_func = marginalized_funcs.cuda_gradient_and_score
        else:
            self._gradient_and_score_func = {
                _Optimizer.Device.AUTO: marginalized_funcs.gradient_and_score,
                _Optimizer.Device.CPU: marginalized_funcs.cython_gradient_and_score
            }[device]
        return self

    @property
    def training_data(self) -> np.ndarray:
        """
        This property returns all the data given to this optimizer to train a new MHN.
        """
        return self._bin_datamatrix


class DUAOptimizer(_Optimizer):
    """
    This optimizer can make use of age information for each state in the data and can learn an MHN from that using
    methods described in Rupp et al.(2021).
    """

    def __init__(self):
        super().__init__()
        # the matrix exponential functions expect an additional parameter eps for the accuracy of the result
        # therefore, we have two separate "gradient_and_score" members, one that takes eps as parameter, and one that
        # does not and which we use in the train() method
        # the eps can be specified as parameter in the train() method
        self.__gradient_and_score_func_with_eps = mat_exp.cython_gradient_and_score
        self._gradient_and_score_func = lambda theta, data: self.__gradient_and_score_func_with_eps(theta, data, 1e-2)
        self._state_ages = None

    def load_data(self, data_matrix: np.ndarray, state_ages: np.ndarray):
        """
        Load training data consisting of a data matrix containing the observed states (rows represent samples
        and columns genes) and an array containing the ages of each state in the data matrix. Thus, the number of ages
        must align with the number of rows in the given data matrix.

        :param data_matrix: two-dimensional numpy array which should have dtype=np.int32
        :param state_ages: one-dimensional numpy array which should have dtype=np.double
        """
        data_matrix = self._preprocess_binary_matrix(data_matrix)
        if data_matrix.shape[0] != state_ages.shape[0]:
            raise ValueError("The number of samples in the data matrix must align with the number of ages given")

        if not np.all(state_ages[:-1] <= state_ages[1:]):
            warnings.warn("The data is not sorted by age yet. This should be no problem, but if you expected your data"
                          "to be sorted by age, something went wrong.")
        self._data = StateAgeContainer(data_matrix, state_ages)
        self._bin_datamatrix = data_matrix
        self._state_ages = state_ages
        return self

    def set_device(self, device: _Optimizer.Device):
        """
        Set the device that should be used for training.

        You have three options:
            Device.AUTO: (default) automatically select the device that is likely to match the data
            Device.CPU:  use the CPU implementations to compute the scores and gradients
            Device.GPU:  use the GPU/CUDA implementations to compute the scores and gradients

        The Device enum is part of this optimizer class.
        """
        super().set_device(device)
        if device == _Optimizer.Device.GPU:
            raise NotImplementedError("There is currently no GPU version available for the DUA algorithm,"
                                      " might be added later")
        else:
            self.__gradient_and_score_func_with_eps = mat_exp.cython_gradient_and_score
        return self

    def train(self, lam: float = 0, eps: float = 1e-2, maxit: int = 5000, trace: bool = False,
              reltol: float = 1e-7, round_result: bool = True) -> model.MHN:
        """
        Use this function to learn a new MHN from the data given to this optimizer.

        :param lam: tuning parameter lambda for the L1 regularization
        :param eps: accuracy of the matrix exponential
        :param maxit: maximum number of training iterations
        :param trace: set to True to print convergence messages (see scipy.optimize.minimize)
        :param reltol: Gradient norm must be less than reltol before successful termination (see "gtol" scipy.optimize.minimize)
        :param round_result: if True, the result is rounded to two decimal places
        :return: trained model
        """
        self._gradient_and_score_func = lambda theta, data: self.__gradient_and_score_func_with_eps(theta, data, eps)
        return super().train(lam, maxit, trace, reltol, round_result)

    @property
    def training_data(self) -> (np.ndarray, np.ndarray):
        """
        This property returns all the data given to this optimizer to train a new MHN.
        """
        return self._bin_datamatrix, self._state_ages
