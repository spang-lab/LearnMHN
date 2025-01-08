"""
This submodule contains classes to represent Mutual Hazard Networks.
"""

# author(s): Y. Linda Hu, Stefan Vocht

from __future__ import annotations

import itertools
import json
import warnings
from math import factorial
from typing import Iterator, Optional, Union
from collections import defaultdict

import matplotlib
import matplotlib.axes
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg.blas import daxpy, dcopy, dscal

from . import utilities
from .training import likelihood_cmhn


def bits_fixed_n(n: int, k: int) -> Iterator[int]:
    """
    Generator over integers whose binary representation has a fixed number of 1s, in lexicographical order.

    From https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation

    Args:
        n (int): How many 1s there should be.
        k (int): How many bits the integer should have.

    Yields:
        Iterator[int]: Integers with the specified binary properties.
    """

    v = int("1" * n, 2)
    stop_no = v << (k - n)
    w = -1
    while w != stop_no:
        t = (v | (v - 1)) + 1
        w = t | ((((t & -t)) // (v & (-v)) >> 1) - 1)
        v, w = w, v
        yield w


class cMHN:
    """
    Represents a classical Mutual Hazard Network (cMHN) (see Schill et al. (2019)).

    Attributes:
        log_theta (np.ndarray): logarithmic values of the theta matrix representing the cMHN
        events (list[str] | None): Names of the events considered by the cMHN.
        meta (dict | None): Metadata for the cMHN, e.g., parameters used to train the model.
    """

    def __init__(self, log_theta: np.array, events: list[str] = None, meta: dict = None):
        """
        Initializes the cMHN with a theta matrix, optional event names, and metadata.

        Args:
            log_theta (np.array): Logarithmic values of the theta matrix representing the cMHN.
            events (list[str], optional): Names of the events considered by the cMHN.
            meta (dict, optional): Metadata for the cMHN, e.g., parameters used to train the model.
        """
        n = log_theta.shape[1]
        self.log_theta = log_theta
        if events is not None and len(events) != n:
            raise ValueError(
                f"the number of events ({len(events)}) does not align with the shape of log_theta ({n}x{n})"
            )
        self.events = events
        self.meta = meta

    def sample_artificial_data(self, sample_num: int, as_dataframe: bool = False) -> np.ndarray | pd.DataFrame:
        """
        Samples artificial data from the cMHN. Use np.random.seed() to make results reproducible.

        Args:
            sample_num (int): Number of samples to generate.
            as_dataframe (bool, optional): Whether to return the data as a pandas DataFrame. Defaults to False.

        Returns:
            np.ndarray | pd.DataFrame: Samples as rows and events as columns, in the specified format.
        """
        art_data = utilities.sample_artificial_data(self.log_theta, sample_num)
        if as_dataframe:
            df = pd.DataFrame(art_data)
            if self.events is not None:
                df.columns = self.events
            return df
        else:
            return art_data

    def sample_trajectories(
        self,
        trajectory_num: int,
        initial_state: np.ndarray | list[str],
        output_event_names: bool = False,
    ) -> tuple[list[list[int | str]], np.ndarray]:
        """
        Simulates event accumulation using the Gillespie algorithm. Use np.random.seed() to make results reproducible.

        Args:
            trajectory_num (int): Number of trajectories to simulate.
            initial_state (np.ndarray | list[str]): Starting state for the trajectories. Can be either a numpy array containing 0s and 1s, where each entry
                                                    represents an event being present (1) or not (0), or a list of strings, where each string is the name of
                                                    an event. The later can only be used if events were specified during creation of the cMHN object.
            output_event_names (bool, optional): Whether to return event names instead of indices. Defaults to False.

        Returns:
            tuple[list[list[int | str]], np.ndarray]: List of trajectories and their observation times.
        """
        if type(initial_state) is np.ndarray:
            initial_state = initial_state.astype(np.int32)
            if initial_state.size != self.log_theta.shape[1]:
                raise ValueError(
                    f"The initial state must be of size {self.log_theta.shape[1]}"
                )
            if not set(initial_state.flatten()).issubset({0, 1}):
                raise ValueError(
                    "The initial state array must only contain 0s and 1s"
                )
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

        trajectory_list, observation_times = utilities.gillespie(
            self.log_theta, initial_state, trajectory_num
        )

        if output_event_names:
            if self.events is None:
                raise ValueError(
                    "output_event_names can only be set to True, if events was set for the cMHN object"
                )
            trajectory_list = list(
                map(
                    lambda trajectory: list(
                        map(lambda event: self.events[event], trajectory)
                    ),
                    trajectory_list,
                )
            )

        return trajectory_list, observation_times

    def compute_marginal_likelihood(self, state: np.ndarray) -> float:
        """
        Computes the likelihood of observing a given state. We consider the observation time to be an exponential random variable with mean 1.

        Args:
            state (np.ndarray): Binary array (dtype=np.int32) representing the presence (1) or absence (0) of events.

        Returns:
            float: Likelihood of the given state.

        Raises:
            ValueError: If the given state array contains anything but 0s and 1s.
        """
        if not set(state.flatten()).issubset({0, 1}):
            raise ValueError("The state array must only contain 0s and 1s")
        mutation_num = np.sum(state)
        nx = 1 << mutation_num
        p0 = np.zeros(nx)
        p0[0] = 1
        p_th = likelihood_cmhn.compute_restricted_inverse(
            self.log_theta, state, p0, False
        )
        return p_th[-1]

    def compute_next_event_probs(
        self,
        state: np.ndarray,
        as_dataframe: bool = False,
        allow_observation: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        """
        Computes probabilities for each event to be the next to occur.

        Args:
            state (np.ndarray): Binary array (dtype=np.int32) representing the presence (1) or absence (0) of events.
            as_dataframe (bool, optional): Whether to return the probabilities as a DataFrame. Defaults to False.
            allow_observation (bool, optional): Whether to include an observation event in the probabilities. Defaults to False.

        Returns:
            np.ndarray | pd.DataFrame: Probabilities for the next event, in the specified format.

        Raises:
            ValueError: If the number of events in state does not align with the number of events modeled by this cMHN object.
        """
        n = self.log_theta.shape[1]
        if n != state.shape[0]:
            raise ValueError(
                f"This cMHN object models {n} events, but state contains {state.shape[0]}"
            )
        if allow_observation:
            observation_rate = self._get_observation_rate(state)
        else:
            observation_rate = 0
        result = utilities.compute_next_event_probs(
            self.log_theta, state, observation_rate
        )
        if allow_observation:
            result = np.concatenate((result, [1 - result.sum()]))
        if not as_dataframe:
            return result
        df = pd.DataFrame(result)
        df.columns = ["PROBS"]
        if self.events is not None:
            df.index = self.events + (
                ["Observation"] if allow_observation else []
            )
        return df

    def _get_observation_rate(self, state: np.ndarray) -> float:
        """
        Calculates the observation rate for a given state.

        Args:
            state (np.ndarray): Current state of events.

        Returns:
            float: Observation rate.
        """
        return 1.0

    def order_likelihood(self, sigma: tuple[int]) -> float:
        """
        Computes the marginal likelihood of an order of events.

        Args:
            sigma (tuple[int]): Tuple of integers where the integers represent the events.

        Returns:
            float: Marginal likelihood of observing sigma.
        """
        events = np.zeros(self.log_theta.shape[0], dtype=np.int32)
        sigma = np.array(sigma)
        events[sigma] = 1
        pos = np.argsort(np.argsort(sigma))
        restr_diag = self.get_restr_diag(state=events)
        return np.exp(
            sum(
                (
                    self.log_theta[x_i, sigma[:n_i]].sum()
                    + self.log_theta[x_i, x_i]
                )
                for n_i, x_i in enumerate(sigma)
            )
        ) / np.prod(
            [
                1 - restr_diag[(1 << pos)[:i].sum()]
                for i in range(len(sigma) + 1)
            ]
        )

    def likeliest_order(self, state: np.array, normalize: bool = False) -> tuple[float, np.array]:
        """
        Returns the likeliest order in which a given state accumulated according to the MHN.

        Args:
            state (np.array): State (binary, dtype int32), shape (n,) with n the number of total
                              events.
            normalize (bool, optional): Whether to normalize among all possible accumulation orders.
                                        Defaults to False.

        Returns:
            tuple[float, np.ndarray]: Likelihood of the likeliest accumulation order and the order itself.
        """
        restr_diag = self.get_restr_diag(state=state)
        log_theta = self.log_theta[state.astype(bool)][:, state.astype(bool)]

        k = state.sum()
        # {state: highest path probability to this state}
        A = {0: 1 / (1 - restr_diag[0])}
        # {state: path with highest probability to this state}
        B = {0: []}
        for i in range(1, k + 1):  # i is the number of events
            A_new = dict()
            B_new = dict()
            for st in bits_fixed_n(n=i, k=k):
                A_new[st] = -1
                state_events = np.array(
                    [i for i in range(k) if (1 << i) | st == st]
                )  # events in state
                for e in state_events:
                    # numerator in Gotovos formula
                    num = np.exp(log_theta[e, state_events].sum())
                    pre_st = st - (1 << e)
                    if A[pre_st] * num > A_new[st]:
                        A_new[st] = A[pre_st] * num
                        B_new[st] = B[pre_st].copy()
                        B_new[st].append(e)
                A_new[st] /= 1 - restr_diag[st]
            A = A_new
            B = B_new
        i = (1 << k) - 1
        if normalize:
            A[i] /= self.compute_marginal_likelihood(state=state)
        return (
            A[i],
            np.arange(self.log_theta.shape[0])[state.astype(bool)][B[i]],
        )

    def m_likeliest_orders(
        self, state: np.array, m: int, normalize: bool = False
    ) -> tuple[np.array, np.array]:
        """
        Returns the m likeliest orders in which a given state accumulated according to the MHN.

        Args:
            state (np.array): State (binary, dtype int32), shape (n,) with n the number of total
                              events.
            m (int): Number of likeliest orders to compute.
            normalize (bool, optional): Whether to normalize among all possible accumulation orders.
                                        Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: Array of likelihoods of the likeliest accumulation order and
            array of the order itself.
        """
        restr_diag = self.get_restr_diag(state=state)
        log_theta = self.log_theta[state.astype(bool)][:, state.astype(bool)]

        k = state.sum()
        # {state: highest path probability to this state}
        A = {0: np.array(1 / (1 - restr_diag[0]))}
        # {state: path with highest probability to this state}
        B = {0: np.empty(0, dtype=int)}
        for i in range(1, k + 1):  # i is the number of events
            _m = min(factorial(i - 1), m)
            A_new = dict()
            B_new = dict()
            for st in bits_fixed_n(n=i, k=k):
                A_new[st] = np.zeros(i * _m)
                B_new[st] = np.zeros((i * _m, i), dtype=int)
                state_events = np.array(
                    [i for i in range(k) if 1 << i | st == st]
                )  # events in state
                for j, e in enumerate(state_events):
                    # numerator in Gotovos formula
                    num = np.exp(log_theta[e, state_events].sum())
                    pre_st = st - (1 << e)
                    A_new[st][j * _m : (j + 1) * _m] = num * A[pre_st]
                    B_new[st][j * _m : (j + 1) * _m, :-1] = B[pre_st]
                    B_new[st][j * _m : (j + 1) * _m, -1] = e
                sorting = A_new[st].argsort()[::-1][:m]
                A_new[st] = A_new[st][sorting]
                B_new[st] = B_new[st][sorting]
                A_new[st] /= 1 - restr_diag[st]
            A = A_new
            B = B_new
        i = (1 << k) - 1
        if normalize:
            A[i] /= self.compute_marginal_likelihood(state=state)
        return (
            A[i],
            (np.arange(self.log_theta.shape[0])[state.astype(bool)])[
                B[i].flatten()
            ].reshape(-1, k),
        )

    def save(self, filename: str):
        """
        Saves the cMHN to a CSV file. Metadata is stored in a separate JSON file if provided.

        Args:
            filename (str): Name of the CSV file. JSON metadata file is named accordingly.
        """
        pd.DataFrame(
            self.log_theta, columns=self.events, index=self.events
        ).to_csv(f"{filename}")
        if self.meta is not None:
            json_serializable_meta = {}
            # check if objects in self.meta are JSON serializable, if not,
            # convert them to a string
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
        Loads a cMHN object from a CSV file.

        Args:
            filename (str): Name of the CSV file.
            events (list[str], optional): List of event names considered by the cMHN. Defaults to None.

        Returns:
            cMHN: Loaded cMHN object.
        """
        df = pd.read_csv(f"{filename}", index_col=0)
        if (
            events is None
            and (
                df.columns
                != pd.Index([str(x) for x in range(len(df.columns))])
            ).any()
        ):
            events = df.columns.to_list()
        try:
            with open(f"{filename[:-4]}_meta.json", "r") as file:
                meta = json.load(file)
        except FileNotFoundError:
            meta = None
        return cls(np.array(df), events=events, meta=meta)

    def get_restr_diag(self, state: np.array) -> np.array:
        """
        Get the diagonal of the state-space-restricted Q_Theta matrix.

        Args:
            state (np.array): State (binary, dtype int32) which should be considered for the
                              state space restriction. Shape (n,) with n the number of total events.

        Returns:
            np.array: Diagonal of the state-space-restricted Q_Theta matrix. Shape (2^k,) with
            k the number of 1s in state
        """
        k = state.sum()
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
                        dscal(
                            n=current_length,
                            a=0,
                            x=subdiag[current_length:],
                            incx=1,
                        )
                    else:
                        dcopy(
                            n=current_length,
                            x=subdiag,
                            incx=1,
                            y=subdiag[current_length:],
                            incy=1,
                        )
                        dscal(
                            n=current_length,
                            a=exp_theta,
                            x=subdiag[current_length:],
                            incx=1,
                        )

                    current_length *= 2

                elif i == j:
                    exp_theta = -np.exp(self.log_theta[i, j])
                    dscal(n=current_length, a=exp_theta, x=subdiag, incx=1)

            # add the subdiagonal to dg
            daxpy(n=nx, a=1, x=subdiag, incx=1, y=diag, incy=1)
        return diag

    def __str__(self):
        """
        Returns a string representation of the object.

        Returns:
            str: String representation including events, log_theta matrix, and metadata.
        """
        if isinstance(self.meta, dict):
            meta_data_string = "\n".join(
                [f"{key}:\n{value}\n" for key, value in self.meta.items()]
            )
        else:
            meta_data_string = "None"
        return (
            f"EVENTS: \n{self.events}\n\n"
            f"THETA IN LOG FORMAT: \n {self.log_theta}\n\n"
            f"ADDITIONAL METADATA: \n\n{meta_data_string}"
        )

    def plot(
        self,
        cmap_thetas: Union[str, matplotlib.colors.Colormap] = "RdBu_r",
        cmap_brs: Union[str, matplotlib.colors.Colormap] = "Greens",
        colorbar: bool = True,
        annot: Union[float, bool] = 0.1,
        ax: Optional[np.arraymatplotlib.axes.Axes] = None,
        logarithmic: bool = True,
    ) -> (
        tuple[
            matplotlib.image.AxesImage,
            matplotlib.image.AxesImage,
            matplotlib.colorbar.Colorbar,
            matplotlib.colorbar.Colorbar,
        ]
        | tuple[matplotlib.image.AxesImage, matplotlib.image.AxesImage]
    ):
        """
        Plots the theta matrix.

        Args:
            cmap_thetas (Union[str, matplotlib.colors.Colormap], optional):
                Colormap to use for thetas. Defaults to "RdBu_r".
            cmap_brs (Union[str, matplotlib.colors.Colormap], optional):
                Colormap to use for the base rates. Defaults to "Greens".
            colorbar (bool, optional):
                Whether to display the colorbars. Defaults to True.
            annot (Union[float, bool], optional):
                If boolean, either all or no annotations are displayed. If numerical, displays
                annotations for all effects greater than this threshold in the logarithmic theta matrix.
                Defaults to 0.1.
            ax (Optional[matplotlib.axes.Axes], optional):
                Matplotlib axes to plot on. Defaults to None.
            logarithmic (bool, optional):
                If set to True, plots the logarithmic theta matrix, else plots the exponential theta matrix.
                Defaults to True.

        Returns:
            tuple[matplotlib.image.AxesImage, matplotlib.image.AxesImage, matplotlib.colorbar.Colorbar, matplotlib.colorbar.Colorbar] | tuple[matplotlib.image.AxesImage, matplotlib.image.AxesImage]:
                If colorbar is True, returns the two heatmaps and the two colorbars. Else, returns only the two axes images.
        """

        # raise warning that the threshold is applied to the logarithmic values
        if isinstance(annot, float) and not logarithmic:
            warnings.warn(
                f"The annotation threshold of {annot} is applied to the logarithmic theta, not the exponential values. "
                + f"thetas with |exp(theta)| < {annot} are hidden."
            )

        # configure basic plot setup
        n_col = 3 if colorbar else 2
        dim_theta_0 = self.log_theta.shape[0]
        dim_theta_1 = self.log_theta.shape[1]
        figsize = (
            dim_theta_1 * 0.35 + (3.2 if colorbar else 1.8),
            dim_theta_0 * 0.35 + 1,
        )
        width_ratios = (
            [4, dim_theta_1 + 6, 3] if colorbar else [4, dim_theta_1 + 3]
        )

        # create axes object if not provided
        if ax is None:
            _, ax = plt.subplots(
                1,
                n_col,
                figsize=figsize,
                width_ratios=width_ratios,
                sharey=True,
                layout="tight",
            )
        else:
            # check if ax is n_col dimensional
            if not isinstance(ax, np.ndarray) or ax.shape != (n_col,):
                # warn and create new axes object
                warnings.warn(
                    f"Provided axes object is not {n_col}-dimensional, creating new axes object"
                )
                _, ax = plt.subplots(
                    1,
                    n_col,
                    figsize=figsize,
                    width_ratios=width_ratios,
                    sharey=True,
                    layout="tight",
                )

        # name axes
        ax_brs, ax_theta = ax[:2]

        # get base rates
        base_rates = np.diag(self.log_theta).reshape(-1, 1)
        if self.__class__ == oMHN:
            base_rates = np.concatenate([base_rates, np.array([[0]])])
        if not logarithmic:
            base_rates = np.exp(base_rates)

        # plot thetas
        if logarithmic:
            _max_th = np.abs(self.log_theta).max()
            theta = self.log_theta.copy()
            np.fill_diagonal(theta, 0)
            im_brs = ax_brs.imshow(base_rates, cmap=cmap_brs)
            im_thetas = ax_theta.imshow(
                theta, cmap=cmap_thetas, vmin=-_max_th, vmax=_max_th
            )
        else:
            _max_th = np.exp(
                np.abs(self.log_theta - np.diag(self.log_theta)).max()
            )
            _max_br = np.exp(np.abs(np.diag(self.log_theta)).max())
            theta = np.exp(self.log_theta)
            np.fill_diagonal(theta, 1)
            im_brs = ax_brs.imshow(
                base_rates,
                norm=colors.LogNorm(vmin=1 / _max_br, vmax=_max_br),
                cmap=cmap_brs,
            )
            im_thetas = ax_theta.imshow(
                theta,
                norm=colors.LogNorm(vmin=1 / _max_th, vmax=_max_th),
                cmap=cmap_thetas,
            )

        # style the plot ticks
        ax_brs.tick_params(length=0)
        ax_brs.set_yticks(
            np.arange(0, dim_theta_0, 1),
            (self.events or list(range(dim_theta_1)))
            + (["Observation"] if dim_theta_0 == dim_theta_1 + 1 else []),
        )
        ax_brs.set_xticks([0], ["Base Rate"])
        ax_brs.tick_params(axis="x", rotation=90)

        ax_theta.tick_params(length=0)
        ax_theta.set_yticks(
            np.arange(0, dim_theta_0, 1),
            (self.events or list(range(dim_theta_1)))
            + (["Observation"] if dim_theta_0 == dim_theta_1 + 1 else []),
        )
        ax_theta.set_xticks(np.arange(0, dim_theta_1, 1), self.events)
        ax_theta.tick_params(axis="x", rotation=90)

        ax_theta.set_ylim((dim_theta_0 - 0.5, -0.5))

        # add annotations
        if annot:
            for i in range(dim_theta_1):
                _ = ax_brs.text(
                    0,
                    i,
                    np.around(base_rates[i, 0], decimals=2),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            if self.__class__ == oMHN:
                _ = ax_brs.text(
                        0,
                        dim_theta_1,
                        np.around(base_rates[-1, 0], decimals=2),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            for i in range(dim_theta_0):
                for j in range(dim_theta_1):
                    if not i == j and (
                        annot is True or np.abs(self.log_theta[i, j]) >= annot
                    ):
                        _ = ax_theta.text(
                            j,
                            i,
                            np.around(theta[i, j], decimals=2),
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

        # add colorbars
        if colorbar:
            ax_cbar = ax[2]
            ax_cbar.axis("off")
            cbar_brs = plt.colorbar(
                im_brs, ax=ax_cbar, orientation="horizontal", aspect=3
            )
            cbar_thetas = plt.colorbar(
                im_thetas, ax=ax_cbar, orientation="horizontal", aspect=3
            )

        if colorbar:
            return im_brs, im_thetas, cbar_thetas, cbar_brs
        else:
            return im_brs, im_thetas

    def plot_orders(
        self,
        *,
        orders: Optional[np.array] = None,
        states: Optional[np.array] = None,
        ax: Optional[np.arraymatplotlib.axes.Axes] = None,
        cmap: Union[str, matplotlib.colors.Colormap] = "hsv",
        markers: list[str] = ["o", "s", "D", "^", "p", "P"],
        names: Optional[list[str]] = None,
    ) -> matplotlib.image.AxesImage:
        """
        Plots a given order of events or, if `states` are provided, plots the most likely order in which the state accumulated its events.

        Args:
            orders (np.ndarray, optional): An array of orders to plot. If None, `states` must be provided to compute the orders. Defaults to None.
            states (np.ndarray, optional): An array of states to compute the orders from. If None, `orders` must be provided. Defaults to None.
            ax (np.arraymatplotlib.axes.Axes, optional): An array of matplotlib axes to plot on. Should have shape (i + 1,) where i is the number
                                                         of orders to plot. If None, new axes will be created. Defaults to None.
            cmap (str | matplotlib.colors.Colormap, optional): The colormap to use for plotting the events. Defaults to "hsv".
            markers (list[str], optional): A list of markers to use for plotting the events. Defaults to ["o", "s", "D", "^", "p", "P"].
            names (list[str], optional): An optional list of names for orders to be plotted as titles. Defaults to None.

        Returns:
            matplotlib.image.AxesImage: The axes with the plotted orders.

        Raises:
            ValueError: If neither `orders` nor `states` are provided, or if both are provided.
        """
        if orders is None:
            if states is None:
                raise ValueError(
                    "Either orders or states must be provided to plot the most likely orders."
                )
            orders = [
                tuple(self.likeliest_order(state=state)[1]) for state in states
            ]
        elif states is not None:
            raise ValueError(
                "Not both orders and states can be provided to plot the most likely orders."
            )

        event_list = self.events or [
            str(i) for i in range(self.log_theta.shape[0])
        ]

        events_used = {event for order in orders for event in order}
        events_used = sorted(events_used)

        if ax is None:
            _, ax = plt.subplots(
                len(orders) + 1,
                figsize=(3, 0.5 * (len(orders) + len(events_used))),
                height_ratios=[2] * len(orders) + [2 * len(events_used)],
            )

        # create the plots
        for i, order in enumerate(orders):
            if len(order) > 1:
                ax[i].plot([0, len(order) - 1], [0, 0], color="grey")
            for j, o in enumerate(order):
                color = plt.get_cmap(cmap)(
                    events_used.index(o) / len(events_used)
                )
                ax[i].scatter(
                    j,
                    0,
                    marker=markers[o % len(markers)],
                    color=color,
                    label=event_list[o],
                    zorder=2,
                    s=50,
                )
            ax[i].axis("off")
            if names is not None:
                ax[i].set_title(names[i])

        # create legend for all markers
        handles, labels = [], []
        for a in ax[:-1]:
            new_handles, new_labels = a.get_legend_handles_labels()
            for handle, label in zip(new_handles, new_labels):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)

        # sort handles and labels by the label's index in the event list
        handles, labels = zip(
            *sorted(zip(handles, labels), key=lambda x: event_list.index(x[1]))
        )

        # add legend of the events used
        ax[-1].legend(handles, labels, loc="center")
        ax[-1].axis("off")

        return ax

    def plot_order_tree(self, orderings: Optional[list[tuple[int]]] = None, states: Optional[np.array] = None, max_event_num: int = 4, min_line_width: int = 1,
                        max_line_width: int = 10, ax: Optional[matplotlib.axes.Axes] = None, inner_circle_radius: float = 2.0,
                        circle_radius_diff: float = 1.0, markers: tuple[str] = ("o", "s", "D", "^", "p", "P", ">"), min_symbol_size: float = 30.,
                        min_number_of_occurrence: int = 3) -> matplotlib.axes.Axes:
        """
        Plots a tree representing the most probable chronological orders of events according to this MHN. Each path from the root of the tree (white circle) to
        a leaf illustrates a possible cancer progression within the given dataset. The symbols along each path denote events whose most probable chronological
        order was derived from this MHN model. Each ordering / state corresponds to a terminal node in the tree or an internal node with a black outline.
        The size of the edges and symbols along a path scale with the total number of patients with that cancer state.

        Args:
            orderings (list[tuple[int]], optional): A list where each element represents an ordering of events that should be added to the tree. The
                elements of each ordering should be the index of the event in this objects events list. If None is given, states is used instead.
            states (np.ndarray, optional): An array of states ((binary, dtype int32), shape (n,) with n the number of total events) to compute the orders from.
                If None, `orderings` must be provided.
            max_event_num (int): Maximum number of events of a single state that should be plotted in the tree. If a state has more that this number of active
                events, only the first active events up until this point are plotted.
            min_line_width (int): Minimum line width of the lines connecting the events in the tree.
            max_line_width (int): Maximum line width of the lines connecting the events in the tree.
            ax (matplotlib.axes.Axes, optional): Axis on which the tree is plotted. If None, a new axis is created.
            inner_circle_radius (float): Distance between the tree root and the first event in the tree.
            circle_radius_diff (float): Difference in radius between the circles on which consecutive events lie on.
            markers (tuple[str]): A list of markers to use for plotting the events. Defaults to ("o", "s", "D", "^", "p", "P", ">").
            min_symbol_size (float): Minimum size of the markers representing events in the tree.
            min_number_of_occurrence (int): Minimum number of occurrence of a state / ordering to be plotted in the tree. Used to avoid clutter.

        Returns:
             matplotlib.axes.Axes: The axis with the plotted tree.
        """

        if orderings is None:
            if states is None:
                raise ValueError(
                    "Either orders or states must be provided to plot the most likely orders."
                )
            orderings = [
                tuple(self.likeliest_order(state=state)[1]) for state in states
            ]

        if ax is None:
            _, ax = plt.subplots()

        circle_num = min(max_event_num, max(map(lambda ordering: len(ordering), orderings)))
        orderings = list(filter(lambda ordering: orderings.count(ordering) >= min_number_of_occurrence, orderings))
        orderings.sort()

        # chronological tree can be seen as a suffix tree
        suffix_tree_root = {"nodes": {}, "leaves": 0, "passed": 0, "is_end": False}
        for ordering in orderings:
            curr_node = suffix_tree_root
            backtracking_nodes = []
            new_leaf = False
            for event in ordering:
                if event not in curr_node["nodes"]:
                    if len(curr_node["nodes"]) > 0:
                        new_leaf = True
                    curr_node["nodes"][event] = {"nodes": {}, "leaves": 1, "passed": 0, "is_end": False}
                curr_node = curr_node["nodes"][event]
                curr_node["passed"] += 1
                backtracking_nodes.append(curr_node)

            curr_node["is_end"] = True
            if new_leaf:
                for node in backtracking_nodes:
                    node["leaves"] += 1

        suffix_tree_root = suffix_tree_root["nodes"]
        event_coordinates = defaultdict(list)
        event_symbol_sizes = defaultdict(list)
        event_symbol_border = defaultdict(list)
        max_passed = max(suffix_tree_root[event]["passed"] for event in suffix_tree_root)

        def recursive_tree_builder(suffix_tree: dict, min_angle: float, max_angle: float, order_idx: int, prev_coordinates: tuple[float, float]):
            """Recursively build the tree by adding the lines and saving the symbol coordinates in event_coordinates."""

            if order_idx > circle_num or len(suffix_tree) == 0:
                return

            circle_radius = inner_circle_radius + order_idx * circle_radius_diff
            curr_angle = min_angle
            total_leaves = sum(suffix_tree[event]["leaves"] for event in suffix_tree)
            for event in suffix_tree:
                node = suffix_tree[event]
                span = node["leaves"] / total_leaves * (max_angle - min_angle)
                symbol_angle = curr_angle + 0.5 * span
                coordinates = np.sin(symbol_angle) * circle_radius, np.cos(symbol_angle) * circle_radius
                event_coordinates[event].append(coordinates)
                linewidth = max(min_line_width, max_line_width * node["passed"] / max_passed)
                event_symbol_sizes[event].append(max(linewidth**2 * np.pi / 2, min_symbol_size))
                event_symbol_border[event].append("black" if node["is_end"] else "white")
                ax.plot(*zip(prev_coordinates, coordinates), marker="", zorder=1,
                        linestyle="-", color="black", linewidth=linewidth)
                recursive_tree_builder(node["nodes"], curr_angle, curr_angle + span, order_idx + 1, coordinates)
                curr_angle += span

        recursive_tree_builder(suffix_tree_root, 0, 2 * np.pi, 0, (0., 0.))
        ax.scatter([0], [0], marker="o", color="white", zorder=2, edgecolors="black", s=max_line_width**2 * np.pi)
        for event, marker in zip(sorted(event_coordinates.keys()), itertools.cycle(markers)):
            event_name = self.events[event] if self.events is not None else str(event)
            ax.scatter(*zip(*event_coordinates[event]), label=event_name, alpha=1, zorder=2, marker=marker,
                       edgecolors=event_symbol_border[event], s=event_symbol_sizes[event])

        ax.axis("off")
        # symbols in legend should have same size (https://stackoverflow.com/questions/24706125/setting-a-fixed-size-for-points-in-legend)
        lgnd = ax.legend()
        for handle in lgnd.legend_handles:
            handle.set_sizes([min_symbol_size])
        return ax


class oMHN(cMHN):
    """
    Represents a Mutual Hazard Network that additionally models the observation event (oMHN) (see Schill et al. (2024)).

    Attributes:
        log_theta (np.ndarray): logarithmic values of the theta matrix representing the oMHN
        events (list[str] | None): Names of the events considered by the cMHN.
        meta (dict | None): Metadata for the oMHN, e.g., parameters used to train the model.
    """

    def sample_artificial_data(
        self, sample_num: int, as_dataframe: bool = False
    ) -> np.ndarray | pd.DataFrame:
        """
        Samples artificial data from the cMHN. Use np.random.seed() to make results reproducible.

        Args:
            sample_num (int): Number of samples to generate.
            as_dataframe (bool, optional): Whether to return the data as a pandas DataFrame. Defaults to False.

        Returns:
            np.ndarray | pd.DataFrame: Samples as rows and events as columns, in the specified format.
        """
        return self.get_equivalent_classical_mhn().sample_artificial_data(
            sample_num, as_dataframe
        )

    def compute_marginal_likelihood(self, state: np.ndarray) -> float:
        """
        Computes the likelihood of observing a given state. We consider the observation time to be an exponential random variable with mean 1.

        Args:
            state (np.ndarray): Binary array (dtype=np.int32) representing the presence (1) or absence (0) of events.

        Returns:
            float: Likelihood of the given state.

        Raises:
            ValueError: If the given state array contains anything but 0s and 1s.
        """
        return self.get_equivalent_classical_mhn().compute_marginal_likelihood(
            state
        )

    def get_equivalent_classical_mhn(self) -> cMHN:
        """
        Converts this oMHN into an equivalent classical cMHN object representing the same distribution.

        Returns:
            cMHN: Equivalent cMHN object.
        """
        n = self.log_theta.shape[1]
        # subtract observation rates from each element in each column
        equivalent_classical_mhn = self.log_theta[:-1] - self.log_theta[-1]
        # undo changes to the diagonal
        equivalent_classical_mhn[range(n), range(n)] += self.log_theta[-1]
        return cMHN(equivalent_classical_mhn, self.events, self.meta)

    def _get_observation_rate(self, state: np.ndarray) -> float:
        """
        Calculates the observation rate for a given state.

        Args:
            state (np.ndarray): Current state of events.

        Returns:
            float: Observation rate.
        """
        return np.exp(np.sum(self.log_theta[-1, state != 0]))

    def save(self, filename: str):
        """
        Saves the oMHN to a CSV file. Metadata is stored in a separate JSON file if provided.

        Args:
            filename (str): Name of the CSV file. JSON metadata file is named accordingly.
        """
        if self.events is None:
            events_and_observation_labels = None
        else:
            events_and_observation_labels = self.events + ["Observation"]
        pd.DataFrame(
            self.log_theta,
            columns=self.events,
            index=events_and_observation_labels,
        ).to_csv(f"{filename}")
        if self.meta is not None:
            json_serializable_meta = {}
            # check if objects in self.meta are JSON serializable, if not,
            # convert them to a string
            for meta_key, meta_value in self.meta.items():
                try:
                    json.dumps(meta_value)
                    json_serializable_meta[meta_key] = meta_value
                except TypeError:
                    json_serializable_meta[meta_key] = str(meta_value)
            with open(f"{filename[:-4]}_meta.json", "w") as file:
                json.dump(json_serializable_meta, file, indent=4)

    def order_likelihood(self, sigma: tuple[int]) -> float:
        """
        Marginal likelihood of an order of events.

        Args:
            sigma (tuple[int]): Tuple of integers where the integers represent the events.

        Returns:
            float: Marginal likelihood of observing sigma.
        """
        return self.get_equivalent_classical_mhn().order_likelihood(sigma)

    def likeliest_order(self, state: np.array,
                        normalize: bool = False) -> tuple[float, np.array]:
        """
        Returns the likeliest order in which a given state accumulated according to the MHN.

        Args:
            state (np.ndarray): State (binary, dtype int32), shape (n,) with n the number of total
                                events.
            normalize (bool, optional): Whether to normalize among all possible accumulation orders.
                                        Defaults to False.

        Returns:
            tuple[float, Any]: Likelihood of the likeliest accumulation order and the order itself.
        """
        return self.get_equivalent_classical_mhn().likeliest_order(
            state, normalize
        )

    def m_likeliest_orders(
        self, state: np.array, m: int, normalize: bool = False
    ) -> tuple[np.array, np.array]:
        """Returns the m likeliest orders in which a given state accumulated according to the MHN.

        Args:
            state (np.ndarray): State (binary, dtype int32), shape (n,) with n the number of total
                                events.
            m (int): Number of likeliest orders to compute.
            normalize (bool, optional): Whether to normalize among all possible accumulation orders.
                                        Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: Array of likelihoods of the likeliest accumulation order and
            array of the order itself.
        """
        return self.get_equivalent_classical_mhn().m_likeliest_orders(
            state, m, normalize
        )
