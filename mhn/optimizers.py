"""
This submodule contains Optimizer classes to learn an MHN from mutation data.
"""
# author(s): Stefan Vocht, Y. Linda Hu, Rudolf Schill

from __future__ import annotations

import warnings
from enum import Enum
import abc

from tqdm.auto import trange


from .ssr.learn_MHN import learn_MHN, reg_state_space_restriction_score, reg_state_space_restriction_gradient
from .ssr.learn_MHN import reg_approximate_score, reg_approximate_gradient


import numpy as np
import pandas as pd

from . import model

from mhn.ssr import regularized_optimization as reg_optim
from .ssr.state_containers import StateContainer, StateAgeContainer
from .ssr.state_containers import create_indep_model
from .ssr.state_space_restriction import CUDAError, cuda_available, CUDA_AVAILABLE
from mhn.ssr import state_space_restriction as marginalized_funcs

import mhn.omega_mhn.with_ssr as omega_funcs


class _Optimizer(abc.ABC):
    """
    This abstract Optimizer class is the base class for the other Optimizer classes and cannot be instantiated alone.
    """


    def __init__(self):
        self._data = None
        self._bin_datamatrix = None
        self._result = None
        self._events = None

        self._init_theta = None
        self.__custom_callback = None

        self.__backup_steps = -1
        self.__backup_filename = None
        self.__backup_always_new_file = False
        self.__backup_current_step = None

        self._gradient_and_score_func = None
        self._regularized_score_func_builder = lambda grad_score_func: \
            reg_optim.build_regularized_score_func(grad_score_func, reg_optim.L1)
        self._regularized_gradient_func_builder = lambda grad_score_func: \
            reg_optim.build_regularized_gradient_func(grad_score_func, reg_optim.L1_)

        self._OutputMHNClass = model.MHN

    def set_init_theta(self, init: np.ndarray):
        """
        Use this method to set a theta as starting point for learning a new MHN. The theta must be in logarithmic form.

        If none is given, the optimization starts with an independence model, where the baseline hazard Theta_ii
        of each event is set to its empirical odds and the hazard ratios (off-diagonal entries) are set to exactly 1.
        """
        self._init_theta = init
        return self

    def get_data_properties(self):
        """
        You can use this method to get some information about the loaded training data, e.g. how many events and samples
        are present in the data, how many events have occurred in a sample on average etc.

        :returns: a dictionary containing information about the data
        """
        if self._bin_datamatrix is None:
            return {}

        total_event_occurrence = np.sum(self._bin_datamatrix, axis=0)
        event_frequencies = total_event_occurrence / self._bin_datamatrix.shape[0]
        event_dataframe = pd.DataFrame.from_dict({
            "Total": total_event_occurrence,
            "Frequency": event_frequencies
        })
        if self._events is not None:
            event_dataframe.index = self._events

        total_events_per_sample = np.sum(self._bin_datamatrix, axis=1)
        return {
            'samples': self._bin_datamatrix.shape[0],
            'events': self._bin_datamatrix.shape[1],
            'occurred events per sample': {
                'mean': np.mean(total_events_per_sample),
                'median': np.median(total_events_per_sample),
                'max': np.max(total_events_per_sample),
                'min': np.min(total_events_per_sample)
            },
            'event statistics': event_dataframe
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

    def set_score_and_gradient_function(self, score_func, gradient_func):
        if not hasattr(score_func, '__call__') or not hasattr(gradient_func, '__call__'):
            raise ValueError(
                "score_func and gradient_func have to be functions!")
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

        self._result = None
        self.__backup_current_step = 0

        if self.__custom_callback is None and self.__backup_steps < 1:
            callback_func = None
        else:
            callback_func = self.__total_callback_func

        self.__result = learn_MHN(self.__data, self.__init, lam, maxit, trace, reltol,
                                  round_result, callback_func, self.__score_func, self.__grad_func)

        self.__backup_current_step = None

        self._result = self._OutputMHNClass(
            log_theta=result.x,
            events=self._events,
            meta={
                "lambda": lam,
                "init": self._init_theta,
                "maxit": maxit,
                "reltol": reltol,
                "score": result.fun,
                "message": result.message,
                "status": result.status,
                "nit": result.nit
            })

        return self._result

    @property
    def result(self) -> model.MHN:
        """
        The resulting MHN after training, same as the return value of the train() method.
        This property mainly exists as a kind of backup to ensure that the result of the training is not lost, if the
        user forgets to save the returned value of the train() method in a variable.
        """
        return self._result

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

    def set_penalty(self, penalty: "_Optimizer.Penalty"):
        """
        Set the penalty that should be used for training.

        You have two options:
            Penalty.L1:          (default) uses the L1 penalty as regularization
            Penalty.SYM_SPARSE:  uses a penalty which induces sparsity and soft symmetry

        The Penalty enum is part of this optimizer class.
        """
        if not isinstance(penalty, _Optimizer.Penalty):
            raise ValueError(f"The given penalty is not an instance of {_Optimizer.Penalty}")
        penalty_score, penalty_gradient = {
            _Optimizer.Penalty.L1: (reg_optim.L1, reg_optim.L1_),
            _Optimizer.Penalty.SYM_SPARSE: (reg_optim.sym_sparse, reg_optim.sym_sparse_deriv)
        }[penalty]
        self._regularized_score_func_builder = lambda grad_score_func: \
            reg_optim.build_regularized_score_func(grad_score_func, penalty_score)
        self._regularized_gradient_func_builder = lambda grad_score_func: \
            reg_optim.build_regularized_gradient_func(grad_score_func, penalty_gradient)
        return self

    class Device(Enum):
        """
        A small Enum which can represent device types.
        """
        AUTO, CPU, GPU = range(3)

    class Penalty(Enum):
        """
        Small Enum which represents penalty functions
        """
        L1, SYM_SPARSE = range(2)


class cMHNOptimizer(_Optimizer):
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
        else:
            self._events = None
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

    def lambda_from_cv(self, lambda_min: float = 0.0001, lambda_max: float = 0.1,
                       steps: int = 9, nfolds: int = 5, lambda_vector: np.ndarray | None = None,
                       show_progressbar: bool = False, return_lambda_scores: bool = False
                       ) -> float | tuple[float, pd.DataFrame]:
        """
        Find the best value for lambda according to the "one standard error rule" through n-fold cross-validation.

        You can specify the lambda values that should be tested in cross-validation by setting the lambda_vector
        parameter accordingly.

        Alternatively, you can specify the minimum, maximum and step size for potential lambda values. This method
        will then create a range of possible lambdas with logarithmic grid-spacing, e.g. (0.0001, 0.0010, 0.0100, 0.1000)
        for lambda_min=0.0001, lambda_max=0.1 and steps=4.

        Use np.random.seed() to make results reproducible.

        :param lambda_min: minimum lambda value that should be tested; this will be ignored if lambda_vector is set
        :param lambda_max: maximum lambda value that should be tested; this will be ignored if lambda_vector is set
        :param steps: number of steps between lambda_min and lambda_max; this will be ignored if lambda_vector is set
        :param nfolds: number of folds used for cross-validation
        :param lambda_vector: a numpy array containing lambda values that should be used for cross-validation
        :param show_progressbar: if True, shows a progressbar during cross-validation
        :param return_lambda_scores: if True, this method will return a tuple containing the best lambda value as well as a Dataframe that contains the mean score of each lambda value tested in cross-validation

        :returns: lambda value that performed best during cross-validation. If return_lambda_scores is set to True, this
        method will return a tuple that contains the best lambda value as well as a Dataframe that contains the mean
        score of each lambda value tested in cross-validation.
        """
        if self._bin_datamatrix is None:
            raise ValueError("You have to load data before you start cross-validation")

        if lambda_vector is None:
            # create a range of possible lambdas with logarithmic grid-spacing
            # e.g. (0.0001,0.0010,0.0100,0.1000) for 4 steps
            lambda_path: np.ndarray = np.exp(np.linspace(np.log(lambda_min + 1e-10), np.log(lambda_max + 1e-10), steps))
        else:
            lambda_path = lambda_vector
            steps = lambda_vector.size

        # shuffle the dataset and cut it into n folds
        shuffled_data = self._bin_datamatrix.copy()
        np.random.shuffle(shuffled_data)
        folds = np.arange(self._bin_datamatrix.shape[0]) % nfolds

        # store the scores for each fold in rows and each lambda in columns
        scores = np.zeros((nfolds, steps))

        # use self.__class__ to make this method also usable for subclasses
        opt = self.__class__()

        # make sure that the same score, gradient and regularization functions are used
        opt._gradient_and_score_func = self._gradient_and_score_func
        opt._regularized_score_func_builder = self._regularized_score_func_builder
        opt._regularized_gradient_func_builder = self._regularized_gradient_func_builder

        disable_progressbar = not show_progressbar

        for j in trange(nfolds, desc="Cross-Validation Folds", position=0, disable=disable_progressbar):
            # designate one of folds as test set and the others as training set
            test_data = shuffled_data[np.where(folds == j)]
            test_data_container = StateContainer(test_data)
            train_data = shuffled_data[np.where(folds != j)]
            opt.load_data_matrix(train_data)

            for i in trange(steps, desc="Lambda Evaluation", position=1, leave=False, disable=disable_progressbar):
                opt.train(lam=lambda_path[i].item())
                theta = opt.result.log_theta
                scores[j, i] = self._gradient_and_score_func(theta, test_data_container)[1]

        # find the best performing lambda with the highest average score over folds
        score_means = np.sum(scores, axis=0) / nfolds
        best_lambda_idx = np.argmax(score_means)

        # choose the actual lambda according to the "one standard error rule"
        standard_error = np.std(scores[:, best_lambda_idx]) / np.sqrt(nfolds)
        threshold = np.max(score_means) - standard_error
        chosen_lambda_idx = np.max(np.argwhere(score_means > threshold))

        if return_lambda_scores:
            score_dataframe = pd.DataFrame.from_dict({
                "Lambda Value": lambda_path,
                "Mean Score": score_means,
                "Standard Error": np.std(scores, axis=0) / np.sqrt(nfolds)
            })
            return lambda_path[chosen_lambda_idx].item(), score_dataframe

        return lambda_path[chosen_lambda_idx].item()

    def set_device(self, device: "cMHNOptimizer.Device"):
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
                _Optimizer.Device.CPU: marginalized_funcs.cpu_gradient_and_score
            }[device]
        return self

    @property
    def training_data(self) -> np.ndarray:
        """
        This property returns all the data given to this optimizer to train a new MHN.
        """
        return self._bin_datamatrix


class oMHNOptimizer(cMHNOptimizer):
    """
    This optimizer models the data with the OmegaMHN.
    """

    def __init__(self):
        super().__init__()
        self._gradient_and_score_func = omega_funcs.gradient_and_score
        self._regularized_score_func_builder = lambda grad_score_func: \
            omega_funcs.build_regularized_score_func(grad_score_func, omega_funcs.L1)
        self._regularized_gradient_func_builder = lambda grad_score_func: \
            omega_funcs.build_regularized_gradient_func(grad_score_func, omega_funcs.L1_)
        self._OutputMHNClass = model.OmegaMHN

    def train(self, lam: float = None, maxit: int = 5000, trace: bool = False,
              reltol: float = 1e-7, round_result: bool = True) -> model.OmegaMHN:
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

        undo_init_theta = False
        if self._init_theta is None:
            undo_init_theta = True
            vanilla_theta = create_indep_model(self._data)
            n = vanilla_theta.shape[0]
            omega_theta = np.zeros((n + 1, n))
            omega_theta[:n] = vanilla_theta
            self._init_theta = omega_theta

        super().train(lam, maxit, trace, reltol, round_result)

        if undo_init_theta:
            self._init_theta = None

        return self.result

    @property
    def result(self) -> model.OmegaMHN:
        """
        The resulting OmegaMHN after training, same as the return value of the train() method.
        This property mainly exists as a kind of backup to ensure that the result of the training is not lost, if the
        user forgets to save the returned value of the train() method in a variable.
        """
        return self._result

    def set_device(self, device: "_Optimizer.Device"):
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

            self._gradient_and_score_func = omega_funcs.cuda_gradient_and_score
        else:
            self._gradient_and_score_func = {
                _Optimizer.Device.AUTO: omega_funcs.gradient_and_score,
                _Optimizer.Device.CPU: omega_funcs.cpu_gradient_and_score
            }[device]
        return self

    def set_penalty(self, penalty: "_Optimizer.Penalty"):
        """
        Set the penalty that should be used for training.

        You have two options:
            Penalty.L1:          (default) uses the L1 penalty as regularization
            Penalty.SYM_SPARSE:  uses a penalty which induces sparsity and soft symmetry

        The Penalty enum is part of this optimizer class.
        """
        if not isinstance(penalty, oMHNOptimizer.Penalty):
            raise ValueError(f"The given penalty is not an instance of {oMHNOptimizer.Penalty}")
        penalty_score, penalty_gradient = {
            oMHNOptimizer.Penalty.L1: (omega_funcs.L1, omega_funcs.L1_),
            oMHNOptimizer.Penalty.SYM_SPARSE: (omega_funcs.sym_sparse, omega_funcs.sym_sparse_deriv)
        }[penalty]
        self._regularized_score_func_builder = lambda grad_score_func: \
            omega_funcs.build_regularized_score_func(grad_score_func, penalty_score)
        self._regularized_gradient_func_builder = lambda grad_score_func: \
            omega_funcs.build_regularized_gradient_func(grad_score_func, penalty_gradient)
        return self
