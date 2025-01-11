"""
This submodule contains Optimizer classes to learn an MHN from mutation data.
"""
# author(s): Stefan Vocht, Y. Linda Hu, Rudolf Schill

from __future__ import annotations

import abc
import warnings
from enum import Enum, auto
from typing import Union

import numpy as np
import pandas as pd
from tqdm.auto import trange

from mhn.training import (likelihood_cmhn, likelihood_omhn, penalties_cmhn,
                          penalties_omhn)
from mhn.training import regularized_optimization as reg_optim

from . import model
from .training.likelihood_cmhn import CUDA_AVAILABLE, CUDAError, cuda_available
from .training.state_containers import StateContainer, create_indep_model


class Device(Enum):
    """
    Enum of device types.

    Attributes:
        AUTO (int): Automatically selects the device based on the number of active events in each sample.
        CPU (int): Executes all computations on the CPU.
        GPU (int): Executes score and gradient computations on the GPU.
    """

    AUTO, CPU, GPU = range(3)


class Penalty(Enum):
    """
    Enumeration of penalty functions.

    Attributes:
        L1 (int): Applies L1 regularization during training.
        L2 (int): Applies L2 regularization during training.
        SYM_SPARSE (int): Induces sparsity and soft symmetry (see Schill et al., 2024).
    """

    L1, L2, SYM_SPARSE = range(3)


class _Optimizer(abc.ABC):
    """
    Abstract Optimizer class serving as a base for other optimizer classes.

    Attributes:
        Device (Device): Enum reference for device types.
        Penalty (Penalty): Enum reference for penalty types.
    """

    # Reference to the external enum (re-export), makes separate import of Enums unnecessary
    Device = Device
    Penalty = Penalty

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
            penalties_cmhn.build_regularized_score_func(
                grad_score_func, penalties_cmhn.l1)
        self._regularized_gradient_func_builder = lambda grad_score_func: \
            penalties_cmhn.build_regularized_gradient_func(
                grad_score_func, penalties_cmhn.l1_)

        self._OutputMHNClass = model.cMHN

    def set_init_theta(self, init: np.ndarray | None) -> _Optimizer:
        """
        Sets the initial theta matrix for learning a new MHN.

        Args:
            init (np.ndarray | None): Initial theta matrix in logarithmic form. If None, uses an independence model where the baseline hazard Theta_ii
                                      of each event is set to its empirical odds and the hazard ratios (off-diagonal entries) are set to exactly 1.

        Returns:
            _Optimizer: The optimizer instance.
        """
        self._init_theta = init
        return self

    def get_data_properties(self) -> dict:
        """
        Retrieves properties of the loaded training data.

        Returns:
            dict: A dictionary with information about the training data, including sample and event statistics.
        """
        if self._bin_datamatrix is None:
            return {}

        total_event_occurrence = np.sum(self._bin_datamatrix, axis=0)
        event_frequencies = total_event_occurrence / \
            self._bin_datamatrix.shape[0]
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

    def set_callback_func(self, callback=None) -> _Optimizer:
        """
        Sets a callback function to be invoked after each iteration of the BFGS algorithm.

        Args:
            callback (Callable): A function that takes a single argument (theta matrix computed in the last iteration). Defaults to None.

        Raises:
            ValueError: If the provided callback is not callable.

        Returns:
            _Optimizer: The optimizer instance.
        """
        if callback is not None and not callable(callback):
            raise ValueError("callback has to be a function!")
        self.__custom_callback = callback
        return self

    def save_progress(self, steps: int = -1, always_new_file: bool = False, filename: str = 'theta_backup.npy') -> _Optimizer:
        """
        Configures periodic saving of training progress.

        Args:
            steps (int): Number of training iterations between progress saves. Defaults to -1 (disabled).
            always_new_file (bool): Whether to save each backup as a new file. Defaults to False.
            filename (str): Name of the backup file. Defaults to 'theta_backup.npy'.

        Returns:
            _Optimizer: The optimizer instance.
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
              reltol: float = 1e-7, round_result: bool = True) -> model.cMHN:
        """
        Trains a new MHN model using the loaded data.

        Args:
            lam (float, optional): Regularization parameter. Defaults to 1/(number of samples).
            maxit (int): Maximum number of training iterations. Defaults to 5000.
            trace (bool): Whether to print convergence messages. Defaults to False.
            reltol (float): Gradient norm tolerance for termination. Defaults to 1e-7. (see "gtol" in scipy.optimize.minimize)
            round_result (bool): Whether to round the result to two decimal places. Defaults to True.

        Returns:
            model.cMHN: The trained MHN model.

        Raises:
            ValueError: If no data has been loaded.
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

        score_func = self._regularized_score_func_builder(
            self._gradient_and_score_func)
        gradient_func = self._regularized_gradient_func_builder(
            self._gradient_and_score_func)

        result = reg_optim.learn_mhn(self._data, score_func, gradient_func, self._init_theta, lam, maxit, trace, reltol,
                                     round_result, callback_func)

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
    def result(self) -> model.cMHN:
        """
        Property for retrieving the training result.

        Returns:
            model.cMHN: The resulting MHN model after training.
        """
        return self._result

    @property
    @abc.abstractmethod
    def training_data(self):
        """
        This property returns all the data given to this optimizer to train a new cMHN.
        """
        pass

    @staticmethod
    def _preprocess_binary_matrix(data_matrix: np.ndarray) -> np.ndarray:
        """
        Preprocesses a binary data matrix to ensure the correct format.

        Args:
            data_matrix (np.ndarray): Input binary matrix.

        Returns:
            np.ndarray: Preprocessed data matrix with dtype set to np.int32.

        Raises:
            ValueError: If the matrix is not two-dimensional or contains values other than 0 and 1.
        """
        if len(data_matrix.shape) != 2:
            raise ValueError("The given data matrix must be two-dimensional")
        # StateContainer only accepts numpy arrays with dtype=np.int32
        if data_matrix.dtype != np.int32:
            data_matrix = data_matrix.astype(dtype=np.int32)
            warnings.warn(
                "The dtype of the given data matrix is changed to np.int32")
        if not set(data_matrix.flatten()).issubset({0, 1}):
            raise ValueError("The data matrix must only contain 0s and 1s")

        return data_matrix

    @abc.abstractmethod
    def set_device(self, device: Device) -> _Optimizer:
        """
        Sets the computational device for training.

        You have three options:
            Device.AUTO: (default) automatically select the device that is likely to match the data
            Device.CPU:  use the CPU implementations to compute the scores and gradients
            Device.GPU:  use the GPU/CUDA implementations to compute the scores and gradients

        The Device enum is part of this optimizer class.

        Args:
            device (Device): The device to use (AUTO, CPU, or GPU).

        Returns:
            _Optimizer: The optimizer instance.

        Raises:
            ValueError: If the given device is not an instance of Device.
        """
        if not isinstance(device, Device):
            raise ValueError(
                f"The given device is not an instance of {Device}")

        return self

    def set_penalty(self, penalty: Penalty):
        """
        Sets the penalty type for training.

        You have three options:
            Penalty.L1:          (default) uses the L1 penalty as regularization
            Penalty.L2:          uses the L2 penalty as regularization
            Penalty.SYM_SPARSE:  uses a penalty which induces sparsity and soft symmetry

        The Penalty enum is part of this optimizer class.

        Args:
            penalty (Penalty): The penalty to use (L1, L2, SYM_SPARSE).

        Returns:
            _Optimizer: The optimizer instance.

        Raises:
            ValueError: If the given penalty is not an instance of Penalty.
        """
        if not isinstance(penalty, Penalty):
            raise ValueError(
                f"The given penalty is not an instance of {_Optimizer.Penalty}")
        penalty_score, penalty_gradient = {
            Penalty.L1: (penalties_cmhn.l1, penalties_cmhn.l1_),
            Penalty.L2: (penalties_cmhn.l2, penalties_cmhn.l2_),
            Penalty.SYM_SPARSE: (
                penalties_cmhn.sym_sparse, penalties_cmhn.sym_sparse_deriv)
        }[penalty]
        self._regularized_score_func_builder = lambda grad_score_func: \
            penalties_cmhn.build_regularized_score_func(
                grad_score_func, penalty_score)
        self._regularized_gradient_func_builder = lambda grad_score_func: \
            penalties_cmhn.build_regularized_gradient_func(
                grad_score_func, penalty_gradient)
        return self


class cMHNOptimizer(_Optimizer):
    """
    Optimizes an cMHN for given cross-sectional data.
    """

    def __init__(self):
        super().__init__()
        self._gradient_and_score_func = likelihood_cmhn.gradient_and_score

    def load_data_matrix(self, data_matrix: np.ndarray | pd.DataFrame):
        """
        Loads mutation data stored in a numpy array or pandas DataFrame.

        Args:
            data_matrix (np.ndarray | pd.DataFrame): Data matrix where rows represent samples and columns represent genes.
                                                     Mutations of genes are represented by 1s, intact genes by 0s.

        Returns:
            cMHNOptimizer: This optimizer object.
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

        Args:
            src (str): Path to the CSV file.
            delimiter (str, optional): Delimiter used in the CSV file (default is ',').
            kwargs: Additional keyword arguments passed to pandas' read_csv() function.

        Returns:
            cMHNOptimizer: This optimizer object.
        """
        df = pd.read_csv(src, delimiter=delimiter, **kwargs)
        self.load_data_matrix(df)
        return self

    def lambda_from_cv(self, lambda_min: float | None = None, lambda_max: float | None = None,
                       steps: int = 9, nfolds: int = 5, lambda_vector: np.ndarray | None = None,
                       show_progressbar: bool = False, return_lambda_scores: bool = False, pick_1se: bool = True,
                       ) -> float | tuple[float, pd.DataFrame]:
        """
        Finds the best value for lambda according to either the maximal average test set likelihood or the
        "one-standard-error-rule" through n-fold cross-validation.

        You can specify the lambda values that should be tested in cross-validation by setting the lambda_vector
        parameter accordingly.

        Alternatively, you can specify the minimum, maximum and step size for potential lambda values. This method
        will then create a range of possible lambdas with logarithmic grid-spacing, e.g. (0.0001, 0.0010, 0.0100, 0.1000)
        for lambda_min=0.0001, lambda_max=0.1 and steps=4.

        If you set neither lambda_vector nor lambda_min and lambda_max, the default range (0.1/#datasamples, 10/#datasamples)
        will be used.

        By default, the largest lambda that performed within one standard error of the best-performing lambda is returned as the
        preferred lambda ("one-standard-error-rule"). When setting pick_1se=False, the function will simply return the best-performing lambda instead.

        Use np.random.seed() to make results reproducible.

        Args:
            lambda_min (float, optional): Minimum lambda value to test. Will be ignored if lambda_vector is set.
            lambda_max (float, optional): Maximum lambda value to test. Will be ignored if lambda_vector is set.
            steps (int, optional): Number of steps between lambda_min and lambda_max. Defaults to 9. Will be ignored if lambda_vector is set.
            nfolds (int, optional): Number of folds for cross-validation. Defaults to 5.
            lambda_vector (np.ndarray, optional): Specific lambda values to test.
            show_progressbar (bool, optional): Whether to show a progress bar during cross-validation. Defaults to False.
            return_lambda_scores (bool, optional): Whether to return lambda scores along with the best lambda. Defaults to False.
            pick_1se (bool, optional): if True (default), applies the one-standard-error-rule to pick the returned lambda value. If False,
                                       returns the best-performing lambda.

        Returns:
            float | tuple[float, pd.DataFrame]: Best lambda value, or, if return_lambda_scores is set to True, a tuple with the best lambda and
                                                a DataFrame containing the mean scores for each lambda.
        """
        if self._bin_datamatrix is None:
            raise ValueError(
                "You have to load data before you start cross-validation")

        if lambda_min is None and lambda_max is not None or lambda_min is not None and lambda_max is None:
            raise ValueError("You have to set both lambda_min and lambda_max, if you want to use them.")

        if lambda_min is None and lambda_max is None:
            # the default lambda value used in train() if lambda is not set
            default_lambda = 1 / self._data.get_data_shape()[0]
            lambda_min = 0.1 * default_lambda
            lambda_max = 10 * default_lambda

        if lambda_vector is None:
            # create a range of possible lambdas with logarithmic grid-spacing
            # e.g. (0.0001,0.0010,0.0100,0.1000) for 4 steps
            lambda_path: np.ndarray = np.exp(np.linspace(
                np.log(lambda_min + 1e-10), np.log(lambda_max + 1e-10), steps))
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
                # make sure that events have not a rate of zero, which can become
                # a problem if the event is present in the test set, in which case
                # this would lead to score = -inf
                minimum_base_rate = -np.log(train_data.shape[0] + 1)  # log(1 / (#samples + 1))
                theta[theta < minimum_base_rate] = minimum_base_rate
                scores[j, i] = self._gradient_and_score_func(
                    theta, test_data_container)[1]

        # find the best performing lambda with the highest average score over folds
        score_means = np.sum(scores, axis=0) / nfolds
        best_lambda_idx = np.argmax(score_means)

        if pick_1se:
            # choose the actual lambda according to the "one standard error rule"
            standard_error = np.std(scores[:, best_lambda_idx]) / np.sqrt(nfolds)
            threshold = np.max(score_means) - standard_error
            chosen_lambda_idx = np.max(np.argwhere(score_means >= threshold))
        else:
            # simply choose the best-performing lambda
            chosen_lambda_idx = best_lambda_idx

        chosen_lambda = lambda_path[chosen_lambda_idx].item()

        if not lambda_path.min() < chosen_lambda < lambda_path.max():
            warnings.warn(
                "Optimal lambda is at a limit (min/max) of the given search range. Consider re-running with adjusted search range.")

        if return_lambda_scores:
            score_dataframe = pd.DataFrame.from_dict({
                "Lambda Value": lambda_path,
                "Mean Score": score_means,
                "Standard Error": np.std(scores, axis=0) / np.sqrt(nfolds)
            })
            return chosen_lambda, score_dataframe

        return chosen_lambda

    def set_device(self, device: Device) -> cMHNOptimizer:
        """
        Sets the computational device for training.

        You have three options:
            Device.AUTO: (default) automatically select the device that is likely to match the data
            Device.CPU:  use the CPU implementations to compute the scores and gradients
            Device.GPU:  use the GPU/CUDA implementations to compute the scores and gradients

        The Device enum is part of this optimizer class.

        Args:
            device (Device): The device to use (AUTO, CPU, or GPU).

        Returns:
            cMHNOptimizer: The optimizer instance.

        Raises:
            ValueError: If the given device is not an instance of Device.
        """
        super().set_device(device)
        if device == Device.GPU:
            if cuda_available() != CUDA_AVAILABLE:
                raise CUDAError(cuda_available())

            self._gradient_and_score_func = likelihood_cmhn.cuda_gradient_and_score
        else:
            self._gradient_and_score_func = {
                Device.AUTO: likelihood_cmhn.gradient_and_score,
                Device.CPU: likelihood_cmhn.cpu_gradient_and_score
            }[device]
        return self

    @property
    def training_data(self) -> np.ndarray:
        """
        Returns the data used to train the cMHN model.

        Returns:
            np.ndarray: The data matrix used for training.
        """
        return self._bin_datamatrix


class oMHNOptimizer(cMHNOptimizer):
    """
    Optimizer for the oMHN model.
    """

    def __init__(self):
        super().__init__()
        self._gradient_and_score_func = likelihood_omhn.gradient_and_score
        self._regularized_score_func_builder = lambda grad_score_func: \
            penalties_omhn.build_regularized_score_func(
                grad_score_func, penalties_omhn.l1)
        self._regularized_gradient_func_builder = lambda grad_score_func: \
            penalties_omhn.build_regularized_gradient_func(
                grad_score_func, penalties_omhn.l1_)
        self._OutputMHNClass = model.oMHN

    def train(self, lam: float = None, maxit: int = 5000, trace: bool = False,
              reltol: float = 1e-7, round_result: bool = True) -> model.oMHN:
        """
        Trains a new oMHN model using the loaded data.

        Args:
            lam (float, optional): Regularization parameter. Defaults to 1/(number of samples).
            maxit (int): Maximum number of training iterations. Defaults to 5000.
            trace (bool): Whether to print convergence messages. Defaults to False.
            reltol (float): Gradient norm tolerance for termination. Defaults to 1e-7. (see "gtol" in scipy.optimize.minimize)
            round_result (bool): Whether to round the result to two decimal places. Defaults to True.

        Returns:
            model.oMHN: The trained MHN model.

        Raises:
            ValueError: If no data has been loaded.
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
            self._result.meta["init"] = None

        return self.result

    @property
    def result(self) -> model.oMHN:
        """
        Property for retrieving the training result.

        Returns:
            model.oMHN: The resulting MHN model after training.
        """
        return self._result

    def set_device(self, device: Device):
        """
        Sets the computational device for training.

        You have three options:
            Device.AUTO: (default) automatically select the device that is likely to match the data
            Device.CPU:  use the CPU implementations to compute the scores and gradients
            Device.GPU:  use the GPU/CUDA implementations to compute the scores and gradients

        The Device enum is part of this optimizer class.

        Args:
            device (Device): The device to use (AUTO, CPU, or GPU).

        Returns:
            _Optimizer: The optimizer instance.

        Raises:
            ValueError: If the given device is not an instance of Device.
        """
        super().set_device(device)
        if device == Device.GPU:
            if cuda_available() != CUDA_AVAILABLE:
                raise CUDAError(cuda_available())

            self._gradient_and_score_func = likelihood_omhn.cuda_gradient_and_score
        else:
            self._gradient_and_score_func = {
                Device.AUTO: likelihood_omhn.gradient_and_score,
                Device.CPU: likelihood_omhn.cpu_gradient_and_score
            }[device]
        return self

    def set_penalty(self, penalty: Penalty):
        """
        Sets the penalty type for training.

        You have three options:
            Penalty.L1:          (default) uses the L1 penalty as regularization
            Penalty.L2:          uses the L2 penalty as regularization
            Penalty.SYM_SPARSE:  uses a penalty which induces sparsity and soft symmetry

        The Penalty enum is part of this optimizer class.

        Args:
            penalty (Penalty): The penalty to use (L1, L2, SYM_SPARSE).

        Returns:
            _Optimizer: The optimizer instance.

        Raises:
            ValueError: If the given penalty is not an instance of Penalty.
        """
        if not isinstance(penalty, oMHNOptimizer.Penalty):
            raise ValueError(
                f"The given penalty is not an instance of {oMHNOptimizer.Penalty}")
        penalty_score, penalty_gradient = {
            Penalty.L1: (penalties_omhn.l1, penalties_omhn.l1_),
            Penalty.L2: (penalties_omhn.l2, penalties_omhn.l2_),
            Penalty.SYM_SPARSE: (
                penalties_omhn.sym_sparse, penalties_omhn.sym_sparse_deriv)
        }[penalty]
        self._regularized_score_func_builder = lambda grad_score_func: \
            penalties_omhn.build_regularized_score_func(
                grad_score_func, penalty_score)
        self._regularized_gradient_func_builder = lambda grad_score_func: \
            penalties_omhn.build_regularized_gradient_func(
                grad_score_func, penalty_gradient)
        return self


class MHNType(Enum):
    """
    Enum representing the types of MHN models that can be trained.

    Attributes:
        cMHN: Classical MHN as proposed by Schill et al. (2019).
        oMHN: MHN with observation bias correction as proposed by Schill et al. (2024).
    """
    # add new types with their Optimizer classes here
    cMHN = cMHNOptimizer
    oMHN = oMHNOptimizer

    def get_optimizer(self) -> Union[oMHNOptimizer, cMHNOptimizer]:
        """ Associates each enum member with its optimizer. """
        return self.value()


class Optimizer:
    """
    A dynamic wrapper for optimizer classes (e.g., oMHNOptimizer, cMHNOptimizer) that
    provides access to all methods and attributes of the wrapped optimizer instance.

    Args:
        mhn_type (MHNType, optional): Type of MHN trained by this optimizer class. Defaults to the most recent type.
    """

    # Reference to the external enum (re-export), makes separate import of MHNType unnecessary
    MHNType = MHNType

    def __new__(
        cls, mhn_type: MHNType = MHNType.oMHN
    ) -> Union[oMHNOptimizer, cMHNOptimizer]:
        if not isinstance(mhn_type, MHNType):
            mhn_type_options = ['MHNType.' + member.name for member in MHNType]
            raise ValueError(
                f"Invalid type {mhn_type}. Must be {', '.join(mhn_type_options[:-1])} or {mhn_type_options[-1]}."
            )
        return mhn_type.get_optimizer()
