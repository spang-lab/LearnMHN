"""
This submodule contains classes to represent Mutual Hazard Networks
"""
# author(s): Y. Linda Hu, Stefan Vocht

from __future__ import annotations

from numpy.core.multiarray import array as array

from cmath import log
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from typing import Union, Optional
import matplotlib
import matplotlib.axes
import matplotlib.colors as colors

from . import utilities
from .training import likelihood_cmhn
import warnings

from scipy.linalg.blas import dcopy, dscal, daxpy, ddot
import json
from math import factorial


def Q_from_log_theta(log_theta: np.typing.ArrayLike, diag: bool = True):
    """This function returns the rate matrix Q_Theta of an MHN.

    Args:
        log_theta (np.typing.ArrayLike): Logarithmic Theta Matrix parametrizing the rate matrix.
        diag (bool, optional): Whether to include the diagonal of the matrix. Defaults to True.

    Returns:
        np.array: Q_Theta
    """
    n = log_theta.shape[0]

    def bi(x: int):
        bi = np.array(list(np.binary_repr(x)), dtype=int)
        ze = np.zeros(n)
        ze[-len(bi):] = bi
        return ze

    Q = np.zeros((1 << n, 1 << n))
    for i in range(1 << n):
        for y, b_y in enumerate(reversed(bi(i))):
            if b_y == 0:
                Q[i + (1 << y)][i] = np.exp(log_theta[y][y] + sum(log_theta[y][x]
                                                                  for x in np.nonzero(bi(i))[0]))
    if diag:
        Q = Q - np.diag(Q.sum(axis=0))

    return Q


class bits_fixed_n:
    """
    Iterator over integers whose binary representation has a fixed number of 1s, in lexicographical order

    :param n: How many 1s there should be
    :param k: How many bits the integer should have
    """

    def __init__(self, n, k):
        self.v = int("1"*n, 2)
        self.stop_no = int("1"*n + "0"*(k-n), 2)
        self.stop = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration
        if self.v == self.stop_no:
            self.stop = True
        t = (self.v | (self.v - 1)) + 1
        w = t | ((((t & -t)) // (self.v & (-self.v)) >> 1) - 1)
        self.v, w = w, self.v
        return w


class cMHN:
    """
    This class represents a classical Mutual Hazard Network.
    """

    def __init__(self, log_theta: np.array, events: list[str] = None, meta: dict = None):
        """
        :param log_theta: logarithmic values of the theta matrix representing the cMHN
        :param events: (optional) list of strings containing the names of the events considered by the cMHN
        :param meta: (optional) dictionary containing metadata for the cMHN, e.g. parameters used to train the model
        """
        n = log_theta.shape[1]
        self.log_theta = log_theta
        if events is not None and len(events) != n:
            raise ValueError(
                f"the number of events ({len(events)}) does not align with the shape of log_theta ({n}x{n})")
        self.events = events
        self.meta = meta

    def sample_artificial_data(self, sample_num: int, as_dataframe: bool = False) -> np.ndarray | pd.DataFrame:
        """
        Returns artificial data sampled from this cMHN. Random values are generated with numpy, use np.random.seed()
        to make results reproducible.

        :param sample_num: number of samples in the generated data
        :param as_dataframe: if True, the data is returned as a pandas DataFrame, else as a numpy matrix

        :returns: array or DataFrame with samples as rows and events as columns
        """
        art_data = utilities.sample_artificial_data(
            self.log_theta, sample_num)
        if as_dataframe:
            df = pd.DataFrame(art_data)
            if self.events is not None:
                df.columns = self.events
            return df
        else:
            return art_data

    def sample_trajectories(self, trajectory_num: int, initial_state: np.ndarray | list[str],
                            output_event_names: bool = False) -> tuple[list[list[int | str]], np.ndarray]:
        """
        Simulates event accumulation using the Gillespie algorithm.

        :param trajectory_num: Number of trajectories sampled by the Gillespie algorithm
        :param initial_state: Initial state from which the trajectories start. Can be either a numpy array containing 0s and 1s, where each entry represents an event being present (1) or not (0),
        or a list of strings, where each string is the name of an event. The later can only be used if events were specified during creation of the cMHN object.
        :param output_event_names: If True, the trajectories are returned as lists containing the event names, else they contain the event indices

        :return: A tuple: first element as a list of trajectories, the second element contains the observation times of each trajectory
        """
        if type(initial_state) is np.ndarray:
            initial_state = initial_state.astype(np.int32)
            if initial_state.size != self.log_theta.shape[1]:
                raise ValueError(
                    f"The initial state must be of size {self.log_theta.shape[1]}")
            if not set(initial_state.flatten()).issubset({0, 1}):
                raise ValueError(
                    "The initial state array must only contain 0s and 1s")
        else:
            init_state_copy = list(initial_state)
            initial_state = np.zeros(self.log_theta.shape[1], dtype=np.int32)
            if len(init_state_copy) != 0 and self.events is None:
                raise RuntimeError(
                    "You can only use event names for the initial state, if event was set during initialization of the cMHN object"
                )

            for event in init_state_copy:
                index = self.events.index(event)
                initial_state[index] = 1

        trajectory_list, observation_times = utilities.gillespie(self.log_theta, initial_state, trajectory_num)

        if output_event_names:
            if self.events is None:
                raise ValueError("output_event_names can only be set to True, if events was set for the cMHN object")
            trajectory_list = list(map(
                lambda trajectory: list(map(
                    lambda event: self.events[event],
                    trajectory
                )),
                trajectory_list
            ))

        return trajectory_list, observation_times

    def compute_marginal_likelihood(self, state: np.ndarray) -> float:
        """
        Computes the likelihood of observing a given state, where we consider the observation time to be an
        exponential random variable with mean 1.

        :param state: a 1d numpy array (dtype=np.int32) containing 0s and 1s, where each entry represents an event being present (1) or not (0)

        :returns: the likelihood of observing the given state according to this cMHN
        """
        if not set(state.flatten()).issubset({0, 1}):
            raise ValueError("The state array must only contain 0s and 1s")
        mutation_num = np.sum(state)
        nx = 1 << mutation_num
        p0 = np.zeros(nx)
        p0[0] = 1
        p_th = likelihood_cmhn.compute_restricted_inverse(
            self.log_theta, state, p0, False)
        return p_th[-1]

    def compute_next_event_probs(self, state: np.ndarray, as_dataframe: bool = False,
                                 allow_observation: bool = False) -> np.ndarray | pd.DataFrame:
        """
        Compute the probability for each event that it will be the next one to occur given the current state.

        :param state: a 1d numpy array (dtype=np.int32) containing 0s and 1s, where each entry represents an event being present (1) or not (0)
        :param as_dataframe: if True, the result is returned as a pandas DataFrame, else as a numpy array
        :param allow_observation: if True, the observation event can happen before any other event -> the probabilities of the remaining events will not add up to 100%

        :returns: array or DataFrame that contains the probability for each event that it will be the next one to occur

        :raise ValueError: if the number of events in state does not align with the number of events modeled by this cMHN object
        """
        n = self.log_theta.shape[1]
        if n != state.shape[0]:
            raise ValueError(
                f"This cMHN object models {n} events, but state contains {state.shape[0]}")
        if allow_observation:
            observation_rate = self._get_observation_rate(state)
        else:
            observation_rate = 0
        result = utilities.compute_next_event_probs(
            self.log_theta, state, observation_rate)
        if not as_dataframe:
            return result
        df = pd.DataFrame(result)
        df.columns = ["PROBS"]
        if self.events is not None:
            df.index = self.events
        return df

    def _get_observation_rate(self, state: np.ndarray) -> float:
        return 1.

        def get_restr_diag(self, events: np.array):
        k = events.sum()
        nx = 1 << k
        n = self.log_theta.shape[0]
        diag = np.zeros(nx)
        subdiag = np.zeros(nx)

        for i in range(n):

            current_length = 1
            subdiag[0] = 1
            # compute the ith subdiagonal of Q
            for j in range(n):
                if state[j]:
                    exp_theta = np.exp(self.log_theta[i, j])
                    if i == j:
                        exp_theta *= -1
                        dscal(n=current_length, a=exp_theta, x=subdiag, incx=1)
                        dscal(n=current_length, a=0,
                              x=subdiag[current_length:], incx=1)
                    else:
                        dcopy(n=current_length, x=subdiag, incx=1,
                              y=subdiag[current_length:], incy=1)
                        dscal(n=current_length, a=exp_theta,
                              x=subdiag[current_length:], incx=1)

                    current_length *= 2

                elif i == j:
                    exp_theta = - np.exp(self.log_theta[i, j])
                    dscal(n=current_length, a=exp_theta, x=subdiag, incx=1)

            # add the subdiagonal to dg
            daxpy(n=nx, a=1, x=subdiag, incx=1, y=diag, incy=1)
        return diag

    def order_likelihood(self, sigma: tuple[int]) -> float:
        """Marginal likelihood of an order of events.

        Args:
            sigma (tuple[int]): Tuple of integers where the integers represent the events. 

        Returns:
            float: Marginal likelihood of observing sigma.
        """
        events = np.zeros(self.log_theta.shape[0], dtype=np.int32)
        events[sigma] = 1
        sigma = np.array(sigma)
        pos = np.argsort(np.argsort(sigma))
        restr_diag = self.get_restr_diag(state=events)
        return np.exp(sum((self.log_theta[x_i, sigma[:n_i]].sum() + self.log_theta[x_i, x_i]) for n_i, x_i in enumerate(sigma))) \
            / np.prod([1 - restr_diag[(1 << pos)[:i].sum()] for i in range(len(sigma) + 1)])

    def likeliest_order(self, state: np.array, normalize: bool = False) -> tuple[float, np.array]:
        """Returns the likeliest order in which a given state accumulated according to the MHN.

        Args:
            state (np.array):  State (binary, dtype int32), shape (n,) with n the number of total
            events.
            normalize (bool, optional): Whether to normalize among all possible accumulation orders.
            Defaults to False.

        Returns:
            tuple[float, np.array]: Likelihood of the likeliest accumulation order and the order itself.  
        """
        restr_diag = self.get_restr_diag(state=state)
        log_theta = self.log_theta[state.astype(bool)][:, state.astype(bool)]

        k = state.sum()
        # {state: highest path probability to this state}
        A = {0: 1/(1-restr_diag[0])}
        # {state: path with highest probability to this state}
        B = {0: []}
        for i in range(1, k+1):         # i is the number of events
            A_new = dict()
            B_new = dict()
            for st in bits_fixed_n(n=i, k=k):
                A_new[st] = -1
                state_events = np.array(
                    [i for i in range(k) if (1 << i) | st == st])  # events in state
                for e in state_events:
                    # numerator in Gotovos formula
                    num = np.exp(log_theta[e, state_events].sum())
                    pre_st = st - (1 << e)
                    if A[pre_st] * num > A_new[st]:
                        A_new[st] = A[pre_st] * num
                        B_new[st] = B[pre_st].copy()
                        B_new[st].append(e)
                A_new[st] /= (1-restr_diag[st])
            A = A_new
            B = B_new
        i = (1 << k) - 1
        if normalize:
            A[i] /= self.compute_marginal_likelihood(state=state)
        return (A[i], np.arange(self.log_theta.shape[0])[state.astype(bool)][B[i]])

    def m_likeliest_orders(self, state: np.array, m: int, normalize: bool = False) -> tuple[np.array, np.array]:
        """Returns the m likeliest orders in which a given state accumulated according to the MHN.

        Args:
            state (np.array):  State (binary, dtype int32), shape (n,) with n the number of total
            events.
            m (int): Number of likeliest orders to compute.
            normalize (bool, optional): Whether to normalize among all possible accumulation orders.
            Defaults to False.

        Returns:
            tuple[np.array, np.array]: Array of likelihoods of the likeliest accumulation order and
            array of the order itself.
        """
        restr_diag = self.get_restr_diag(state=state)
        log_theta = self.log_theta[state.astype(bool)][:, state.astype(bool)]

        k = state.sum()
        # {state: highest path probability to this state}
        A = {0: np.array(1/(1-restr_diag[0]))}
        # {state: path with highest probability to this state}
        B = {0: np.empty(0, dtype=int)}
        for i in range(1, k+1):                     # i is the number of events
            _m = min(factorial(i - 1), m)
            A_new = dict()
            B_new = dict()
            for st in bits_fixed_n(n=i, k=k):
                A_new[st] = np.zeros(i * _m)
                B_new[st] = np.zeros((i * _m, i), dtype=int)
                state_events = np.array(
                    [i for i in range(k) if 1 << i | st == st])  # events in state
                for j, e in enumerate(state_events):
                    # numerator in Gotovos formula
                    num = np.exp(log_theta[e, state_events].sum())
                    pre_st = st - (1 << e)
                    A_new[st][j * _m: (j + 1) * _m] = num * A[pre_st]
                    B_new[st][j * _m: (j + 1) * _m, :-1] = B[pre_st]
                    B_new[st][j * _m: (j + 1) * _m, -1] = e
                sorting = A_new[st].argsort()[::-1][:m]
                A_new[st] = A_new[st][sorting]
                B_new[st] = B_new[st][sorting]
                A_new[st] /= (1-restr_diag[st])
            A = A_new
            B = B_new
        i = (1 << k) - 1
        if normalize:
            A[i] /= self.compute_marginal_likelihood(state=state)
        return (A[i], (np.arange(self.log_theta.shape[0])[state.astype(bool)])[B[i].flatten()].reshape(-1, k))

    def save(self, filename: str):
        """
        Save the cMHN in a CSV file. If metadata is given, it will be stored in a separate JSON file.

        :param filename: name of the CSV file, JSON will be named accordingly
        """
        pd.DataFrame(self.log_theta, columns=self.events,
                     index=self.events).to_csv(f"{filename}")
        if self.meta is not None:
            json_serializable_meta = {}
            # check if objects in self.meta are JSON serializable, if not, convert them to a string
            for meta_key, meta_value in self.meta.items():
                try:
                    json.dumps(meta_value)
                    json_serializable_meta[meta_key] = meta_value
                except TypeError:
                    json_serializable_meta[meta_key] = str(meta_value)
            with open(f"{filename[:-4]}_meta.json", "w") as file:
                json.dump(json_serializable_meta, file, indent=4)

    @classmethod
    def load(cls, filename: str, events: list[str] = None) -> cMHN:
        """
        Load an cMHN object from a CSV file.

        :param filename: name of the CSV file
        :param events: list of strings containing the names of the events considered by the cMHN

        :returns: cMHN object
        """
        df = pd.read_csv(f"{filename}", index_col=0)
        if events is None and (df.columns != pd.Index([str(x) for x in range(len(df.columns))])).any():
            events = df.columns.to_list()
        try:
            with open(f"{filename[:-4]}_meta.json", "r") as file:
                meta = json.load(file)
        except FileNotFoundError:
            meta = None
        return cls(np.array(df), events=events, meta=meta)

    def __str__(self):
        if isinstance(self.meta, dict):
            meta_data_string = '\n'.join(
                [f'{key}:\n{value}\n' for key, value in self.meta.items()])
        else:
            meta_data_string = "None"
        return f"EVENTS: \n{self.events}\n\n" \
               f"THETA IN LOG FORMAT: \n {self.log_theta}\n\n" \
               f"ADDITIONAL METADATA: \n\n{meta_data_string}"

    def plot(
            self,
            cmap: Union[str, matplotlib.colors.Colormap] = "RdBu_r",
            colorbar: bool = True,
            annot: Union[float, bool] = 0.1,
            ax: Optional[matplotlib.axes.Axes] = None,
            logarithmic: bool = True
    ) -> None:
        """
        Plots the theta matrix.

        Args:
            cmap (Union[str, matplotlib.colors.Colormap], optional):
                Colormap to use. Defaults to "RdBu_r".
            colorbar (bool, optional):
                Whether to display a colorbar. Defaults to True.
            annot (Union[float, bool], optional):
                If boolean, either all or no annotations are displayed. If numerical, displays
                annotations for all effects greater than this threshold in the logarithmic theta matrix.
                Defaults to 0.1.
            ax (Optional[matplotlib.axes.Axes], optional):
                Matplotlib axes to plot on. Defaults to None.
            logarithmic (bool, optional):
                If set to True, plots the logarithmic theta matrix, else plots the exponential theta matrix.
                Defaults to True.
        """

        if ax is None:
            _, ax = plt.subplots(ncols=2, figsize=(10, 8))
        else:
            # check if ax is 2 dimensional
            if not isinstance(ax, np.ndarray) or ax.shape != (1, 2):
                # warn and create new axes object
                warnings.warn(
                    "Provided axes object is not 2-dimensional, creating new axes object")
                _, ax = plt.subplots(ncols=2, figsize=(10, 8))

        if logarithmic:
            _max = np.abs(self.log_theta).max()
            theta = self.log_theta.copy()
            np.fill_diagonal(theta, 0)
            im = ax[0].imshow(
                self.log_theta,
                cmap=cmap,
                vmin=-_max, vmax=_max)
        else:
            _max = np.abs(self.log_theta).max()
            _max = np.exp(_max)
            theta = np.around(np.exp(self.log_theta), decimals=2)
            np.fill_diagonal(theta, 0)
            im = ax[0].imshow(
                theta,
                norm=colors.LogNorm(vmin=1 / _max, vmax=_max),
                cmap=cmap)
        if colorbar:
            cbar_0 = plt.colorbar(im, ax=ax)
            if not logarithmic:
                cbar_0.minorticks_off()
                ticks = np.exp(np.linspace(np.log(1 / _max), np.log(_max), 9))
                labels = [f'{t:.1e}'[:-3] for t in ticks]
                cbar_0.set_ticks(
                    ticks, labels=labels
                )
        ax[0].tick_params(length=0)
        ax[0].set_yticks(
            np.arange(0, self.log_theta.shape[0], 1),
            (self.events or list(range(self.log_theta.shape[1]))) +
            (["Observation"] if self.log_theta.shape[0] == self.log_theta.shape[1] + 1 else []))
        ax[0].set_xticks(
            np.arange(0, self.log_theta.shape[1], 1),
            self.events)
        ax[0].tick_params(axis="x", rotation=90)
        if annot:
            for i in range(self.log_theta.shape[0]):
                for j in range(self.log_theta.shape[1]):
                    if annot is True or np.abs(self.log_theta[i, j]) >= annot:
                        text = ax[0].text(j, i, theta[i, j],
                                          ha="center", va="center")


class oMHN(cMHN):
    """
    This class represents an oMHN.
    """

    def sample_artificial_data(self, sample_num: int, as_dataframe: bool = False) -> np.ndarray | pd.DataFrame:
        """
        Returns artificial data sampled from this oMHN. Random values are generated with numpy, use np.random.seed()
        to make results reproducible.

        :param sample_num: number of samples in the generated data
        :param as_dataframe: if True, the data is returned as a pandas DataFrame, else numpy matrix

        :returns: array or DataFrame with samples as rows and events as columns
        """
        return self.get_equivalent_classical_mhn().sample_artificial_data(sample_num, as_dataframe)

    def compute_marginal_likelihood(self, state: np.ndarray) -> float:
        """
        Computes the likelihood of observing a given state, where we consider the observation time to be an
        exponential random variable with mean 1.

        :param state: a 1d numpy array (dtype=np.int32) containing 0s and 1s, where each entry represents an event being present (1) or not (0)

        :returns: the likelihood of observing the given state according to this oMHN
        """
        return self.get_equivalent_classical_mhn().compute_marginal_likelihood(state)

    def get_equivalent_classical_mhn(self) -> cMHN:
        """
        This method returns a classical cMHN object that represents the same distribution as this oMHN object.

        :returns: classical cMHN object representing the same distribution as this oMHN object
        """
        n = self.log_theta.shape[1]
        # subtract observation rates from each element in each column
        equivalent_classical_mhn = self.log_theta[:-1] - self.log_theta[-1]
        # undo changes to the diagonal
        equivalent_classical_mhn[range(n), range(n)] += self.log_theta[-1]
        return cMHN(equivalent_classical_mhn, self.events, self.meta)

    def _get_observation_rate(self, state: np.ndarray) -> float:
        return np.exp(np.sum(self.log_theta[-1, state != 0]))

    def save(self, filename: str):
        """
        Save the oMHN in a CSV file. If metadata is given, it will be stored in a separate JSON file.

        :param filename: name of the CSV file, JSON will be named accordingly
        """
        if self.events is None:
            events_and_observation_labels = None
        else:
            events_and_observation_labels = self.events + ["Observation"]
        pd.DataFrame(self.log_theta, columns=self.events,
                     index=events_and_observation_labels).to_csv(f"{filename}")
        if self.meta is not None:
            json_serializable_meta = {}
            # check if objects in self.meta are JSON serializable, if not, convert them to a string
            for meta_key, meta_value in self.meta.items():
                try:
                    json.dumps(meta_value)
                    json_serializable_meta[meta_key] = meta_value
                except TypeError:
                    json_serializable_meta[meta_key] = str(meta_value)
            with open(f"{filename[:-4]}_meta.json", "w") as file:
                json.dump(json_serializable_meta, file, indent=4)

    def order_likelihood(self, sigma: tuple[int]) -> float:
        """Marginal likelihood of an order of events.

        Args:
            sigma (tuple[int]): Tuple of integers where the integers represent the events. 

        Returns:
            float: Marginal likelihood of observing sigma.
        """
        return self.get_equivalent_classical_mhn().order_likelihood(sigma)

    def likeliest_order(self, state: np.array, normalize: bool = False) -> tuple[float, np.array]:
        """Returns the likeliest order in which a given state accumulated according to the MHN.

        Args:
            state (np.array):  State (binary, dtype int32), shape (n,) with n the number of total
            events.
            normalize (bool, optional): Whether to normalize among all possible accumulation orders.
            Defaults to False.

        Returns:
            tuple[float, Any]: Likelihood of the likeliest accumulation order and the order itself.  
        """
        return self.get_equivalent_classical_mhn().likeliest_order(state, normalize)

    def m_likeliest_orders(self, state: np.array, m: int, normalize: bool = False) -> tuple[np.array, np.array]:
        """Returns the m likeliest orders in which a given state accumulated according to the MHN.

        Args:
            state (np.array):  State (binary, dtype int32), shape (n,) with n the number of total
            events.
            m (int): Number of likeliest orders to compute.
            normalize (bool, optional): Whether to normalize among all possible accumulation orders.
            Defaults to False.

        Returns:
            tuple[np.array, np.array]: Array of likelihoods of the likeliest accumulation order and
            array of the order itself.
        """
        return self.get_equivalent_classical_mhn().m_likeliest_orders(state, m, normalize)
