"""
This submodule contains classes to represent Mutual Hazard Networks
"""
# author(s): Y. Linda Hu, Stefan Vocht

from __future__ import annotations

from .original import Likelihood
from .ssr import state_space_restriction

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from typing import Union, Optional
import matplotlib
import matplotlib.axes
import matplotlib.colors as colors


from scipy.linalg.blas import dcopy, dscal, daxpy, ddot


class bits_fixed_n:
    """
    Iterator over integers whose binary representation has a fixed number, in lexicographical order

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


class MHN:
    """
    This class represents a classical Mutual Hazard Network.
    """

    def __init__(self, log_theta: np.array, events: list[str] = None, meta: dict = None):
        """
        :param log_theta: logarithmic values of the theta matrix representing the MHN
        :param events: (optional) list of strings containing the names of the events considered by the MHN
        :param meta: (optional) dictionary containing metadata for the MHN, e.g. parameters used to train the model
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
        Returns artificial data sampled from this MHN. Random values are generated with numpy, use np.random.seed()
        to make results reproducible.

        :param sample_num: number of samples in the generated data
        :param as_dataframe: if True, the data is returned as a pandas DataFrame, else as a numpy matrix

        :returns: array or DataFrame with samples as rows and events as columns
        """
        art_data = Likelihood.sample_artificial_data(
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
        or a list of strings, where each string is the name of an event. The later can only be used if events were specified during creation of the MHN object.
        :param output_event_names: If True, the trajectories are returned as lists containing the event names, else they contain the event indices

        :return: A tuple: first element as a list of trajectories, the second element contains the observation times of each trajectory
        """
        if type(initial_state) is np.ndarray:
            initial_state = initial_state.astype(np.int32)
            if initial_state.size != self.log_theta.shape[1]:
                raise ValueError(f"The initial state must be of size {self.log_theta.shape[1]}")
            if not set(initial_state.flatten()).issubset({0, 1}):
                raise ValueError("The initial state array must only contain 0s and 1s")
        else:
            init_state_copy = list(initial_state)
            initial_state = np.zeros(self.log_theta.shape[1], dtype=np.int32)
            if len(init_state_copy) != 0 and self.events is None:
                raise RuntimeError(
                    "You can only use event names for the initial state, if event was set during initialization of the MHN object"
                )

            for event in init_state_copy:
                index = self.events.index(event)
                initial_state[index] = 1

        trajectory_list, observation_times = Likelihood.gillespie(self.log_theta, initial_state, trajectory_num)

        if output_event_names:
            if self.events is None:
                raise ValueError("output_event_names can only be set to True, if events was set for the MHN object")
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

        :returns: the likelihood of observing the given state according to this MHN
        """
        if not set(state.flatten()).issubset({0, 1}):
            raise ValueError("The state array must only contain 0s and 1s")
        mutation_num = np.sum(state)
        nx = 1 << mutation_num
        p0 = np.zeros(nx)
        p0[0] = 1
        p_th = state_space_restriction.compute_restricted_inverse(
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

        :raise ValueError: if the number of events in state does not align with the number of events modeled by this MHN object
        """
        n = self.log_theta.shape[1]
        if n != state.shape[0]:
            raise ValueError(
                f"This MHN object models {n} events, but state contains {state.shape[0]}")
        if allow_observation:
            observation_rate = self._get_observation_rate(state)
        else:
            observation_rate = 0
        result = Likelihood.compute_next_event_probs(
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

    def save(self, filename: str):
        """
        Save the MHN in a CSV file. If metadata is given, it will be stored in a separate JSON file.

        :param filename: name of the CSV file without(!) the '.csv', JSON will be named accordingly
        """
        pd.DataFrame(self.log_theta, columns=self.events,
                     index=self.events).to_csv(f"{filename}.csv")
        if self.meta is not None:
            json_serializable_meta = {}
            # check if objects in self.meta are JSON serializable, if not, convert them to a string
            for meta_key, meta_value in self.meta.items():
                try:
                    json.dumps(meta_value)
                    json_serializable_meta[meta_key] = meta_value
                except TypeError:
                    json_serializable_meta[meta_key] = str(meta_value)
            with open(f"{filename}_meta.json", "x") as file:
                json.dump(json_serializable_meta, file, indent=4)

    @classmethod
    def load(cls, filename: str, events: list[str] = None) -> MHN:
        """
        Load an MHN object from a CSV file.

        :param filename: name of the CSV file without(!) the '.csv'
        :param events: list of strings containing the names of the events considered by the MHN

        :returns: MHN object
        """
        df = pd.read_csv(f"{filename}.csv", index_col=0)
        if events is None and (df.columns != pd.Index([str(x) for x in range(len(df.columns))])).any():
            events = df.columns.to_list()
        try:
            with open(f"{filename}_meta.json", "r") as file:
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
            _, ax = plt.subplots()

        if logarithmic:
            _max = np.abs(self.log_theta).max()
            theta = self.log_theta
            im = ax.imshow(
                self.log_theta,
                cmap=cmap,
                vmin=-_max, vmax=_max)
        else:
            _max = np.abs(self.log_theta).max()
            _max = np.exp(_max)
            theta = np.around(np.exp(self.log_theta), decimals=2)
            im = ax.imshow(
                theta,
                norm=colors.LogNorm(vmin=1 / _max, vmax=_max),
                cmap=cmap)
        if colorbar:
            cbar = plt.colorbar(im, ax=ax)
            if not logarithmic:
                cbar.minorticks_off()
                ticks = np.exp(np.linspace(np.log(1 / _max), np.log(_max), 9))
                labels = [f'{t:.1e}'[:-3] for t in ticks]
                cbar.set_ticks(
                    ticks, labels=labels
                )
        ax.tick_params(length=0)
        ax.set_yticks(
            np.arange(0, self.log_theta.shape[0], 1),
            (self.events or list(range(self.log_theta.shape[1]))) +
            (["Observation"] if self.log_theta.shape[0] == self.log_theta.shape[1] + 1 else []))
        ax.set_xticks(
            np.arange(0, self.log_theta.shape[1], 1),
            self.events)
        ax.tick_params(axis="x", rotation=90)
        if annot:
            for i in range(self.log_theta.shape[0]):
                for j in range(self.log_theta.shape[1]):
                    if annot is True or np.abs(self.log_theta[i, j]) >= annot:
                        text = ax.text(j, i, theta[i, j],
                                       ha="center", va="center")


class OmegaMHN(MHN):
    """
    This class represents an OmegaMHN.
    """

    def sample_artificial_data(self, sample_num: int, as_dataframe: bool = False) -> np.ndarray | pd.DataFrame:
        """
        Returns artificial data sampled from this MHN. Random values are generated with numpy, use np.random.seed()
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

        :returns: the likelihood of observing the given state according to this MHN
        """
        return self.get_equivalent_classical_mhn().compute_marginal_likelihood(state)

    def get_equivalent_classical_mhn(self) -> MHN:
        """
        This method returns a classical MHN object that represents the same distribution as this OmegaMHN object.

        :returns: classical MHN object representing the same distribution as this OmegaMHN object
        """
        n = self.log_theta.shape[1]
        # subtract observation rates from each element in each column
        equivalent_classical_mhn = self.log_theta[:-1] - self.log_theta[-1]
        # undo changes to the diagonal
        equivalent_classical_mhn[range(n), range(n)] += self.log_theta[-1]
        return MHN(equivalent_classical_mhn, self.events, self.meta)

    def _get_observation_rate(self, state: np.ndarray) -> float:
        return np.exp(np.sum(self.log_theta[-1, state != 0]))

    def save(self, filename: str):
        """
        Save the MHN in a CSV file. If metadata is given, it will be stored in a separate JSON file.

        :param filename: name of the CSV file without(!) the '.csv', JSON will be named accordingly
        """
        if self.events is None:
            events_and_observation_labels = None
        else:
            events_and_observation_labels = self.events + ["Observation"]
        pd.DataFrame(self.log_theta, columns=self.events,
                     index=events_and_observation_labels).to_csv(f"{filename}.csv")
        if self.meta is not None:
            json_serializable_meta = {}
            # check if objects in self.meta are JSON serializable, if not, convert them to a string
            for meta_key, meta_value in self.meta.items():
                try:
                    json.dumps(meta_value)
                    json_serializable_meta[meta_key] = meta_value
                except TypeError:
                    json_serializable_meta[meta_key] = str(meta_value)
            with open(f"{filename}_meta.json", "x") as file:
                json.dump(json_serializable_meta, file, indent=4)

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
                if events[j]:
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

    def likeliest_order(self, events: np.array):

        restr_diag = self.get_restr_diag(events=events)

        k = events.sum()
        A = {0: 1/(1-restr_diag[0])}
        B = {0: []}
        for i in range(1, k+1):
            A_new = dict()
            B_new = dict()
            for state in bits_fixed_n(n=i, k=k):
                A_new[state] = -1
                state_bin_pos = np.nonzero(
                    np.flip(np.array(list(np.binary_repr(state)), dtype=int)))[0]
                for pos in state_bin_pos:
                    num = np.exp(self.log_theta[pos, state_bin_pos].sum())
                    pre_state = state - (1 << pos)
                    if A[pre_state] * num > A_new[state]:
                        A_new[state] = A[pre_state] * num
                        B_new[state] = B[pre_state].copy()
                        B_new[state].append(pos)
                A_new[state] /= (1-restr_diag[state])
            A = A_new
            B = B_new

        return (A, B)
