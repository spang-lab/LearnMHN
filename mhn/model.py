# by Y. Linda Hu

import numpy as np
import pandas as pd
from scipy.linalg.blas import dcopy, dscal, daxpy, ddot


class bits_fixed_n:

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
    This class represents the Mutual Hazard Network
    """

    def __init__(self, log_theta: np.array, events: list[str] = None):

        self.log_theta = log_theta
        self.events = events

    def save(self, filename: str):
        pd.DataFrame(self.log_theta, columns=self.events,
                     index=self.events).to_csv(filename)

    @classmethod
    def load(cls, filename: str, events: list[str] = None):
        """
        :param filename: path to the CSV file
        """
        df = pd.read_csv(filename, index_col=0)
        if events is None and (df.columns != pd.Index([str(x) for x in range(len(df.columns))])).any():
            events = df.columns.to_list()
        return cls(np.array(df), events=events)

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

    # def likeliest_order(events: np.array):

    #     restr_diag =

    #     k = events.sum()
    #     A = {0:1}
    #     B = {0:[]}
    #     for i in range(k):
    #         for state in bits_fixed_n:

    #             for bit in range(k):
    #                 if (1 << bit) & state:
    #                     if S * A[state - (1 << bit)] > A_new[state]:
    #                         A_new[state] = S * A[state - (1 << bit)]
    #                         B_new[state] = B[state - (1 << bit)][i] = bit
    #         A = A_new
    #         B = B_new
