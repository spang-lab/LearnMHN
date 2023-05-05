"""
This submodule contains a class to represent Mutual Hazard Networks
"""
# author(s): Y. Linda Hu, Stefan Vocht

from __future__ import annotations

from .original import Likelihood
from .ssr import state_space_restriction

import numpy as np
import pandas as pd
import json


from scipy.linalg.blas import dcopy, dscal, daxpy
from math import factorial


def Q_from_log_theta(log_theta, diag=True):

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


class MHN:
    """
    This class represents the Mutual Hazard Network.
    """

    def __init__(self, log_theta: np.array, events: list[str] = None, meta: dict = None):
        """
        :param log_theta: logarithmic values of the theta matrix representing the MHN
        :param events: (optional) list of strings containing the names of the events considered by the MHN
        :param meta: (optional) dictionary containing metadata for the MHN, e.g. parameters used to train the model
        """

        self.log_theta = log_theta
        self.events = events
        self.meta = meta

    def sample_artificial_data(self, sample_num: int, as_dataframe: bool = False) -> np.ndarray | pd.DataFrame:
        """
        Returns artificial data sampled from this MHN. Random values are generated with numpy, use np.random.seed()
        to make results reproducible.

        :param sample_num: number of samples in the generated data
        :param as_dataframe: if True, the data is returned as a pandas DataFrame, else numpy matrix

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

    def save(self, filename: str):
        """
        Save the MHN in a CSV file. If metadata is given, it will be stored in a separate JSON file.

        :param filename: name of the CSV file without(!) the '.csv', JSON will be named accordingly
        """
        pd.DataFrame(self.log_theta, columns=self.events,
                     index=self.events).to_csv(f"{filename}.csv")
        if self.meta is not None:
            with open(f"{filename}_meta.json", "x") as file:
                json.dump(self.meta, file, indent=4)

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

    def get_restr_diag(self, state: np.array):
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

    def order_prob(self, sigma):

        events = np.zeros(self.log_theta.shape[0], dtype=np.int32)
        events[sigma] = 1
        sigma = np.array(sigma)
        pos = np.argsort(np.argsort(sigma))
        restr_diag = self.get_restr_diag(state=events)
        return np.exp(sum((self.log_theta[x_i, sigma[:n_i]].sum() + self.log_theta[x_i, x_i]) for n_i, x_i in enumerate(sigma))) \
            / np.prod([1 - restr_diag[(1 << pos)[:i].sum()] for i in range(len(sigma) + 1)])

    def likeliest_order(self, state: np.array):

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
        return (A[i], np.arange(self.log_theta.shape[0])[state.astype(bool)][B[i]])

    def m_likeliest_orders(self, state: np.array, m: int):

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
        return (A[i], (np.arange(self.log_theta.shape[0])[state.astype(bool)])[B[i].flatten()].reshape(-1, k))


if __name__ == "__main__":
    mhn = MHN.load("../likeliestorder/data/breast/log_theta")
    n = mhn.log_theta.shape[0]
    state = np.zeros(n, dtype=int)
    state[:3] = 1
    mhn.m_likeliest_orders(state, m=4)
    
