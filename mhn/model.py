# by Y. Linda Hu

import numpy as np
import pandas as pd
from scipy.linalg.blas import dcopy, dscal, daxpy, ddot
import json


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
    This class represents the Mutual Hazard Network
    """

    def __init__(self, log_theta: np.array, events: list[str] = None, meta: dict = None):

        self.log_theta = log_theta
        self.events = events
        self.meta = meta

    def save(self, filename: str):
        """
        Save the MHN file
        """
        pd.DataFrame(self.log_theta, columns=self.events,
                     index=self.events).to_csv(f"{filename}.csv")
        if self.meta is not None:
            with open(f"{filename}_meta.json", "x") as file:
                json.dump(self.meta, file, indent=4)

    @classmethod
    def load(cls, filename: str, events: list[str] = None):
        """
        :param filename: path to the CSV file
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

### THIS IS YIELDING WEIRD RESULTS, CHECK AGAIN

    def mcmc_sampling(self, events: np.array, n_samples: int=50, burn_in: float=0.2):
                
        
        restr_diag = self.get_restr_diag(events=events)
        mutation_num = events.sum()  

        def proposal():
            S = np.nonzero(events)[0].tolist()
            evs = []
            p = 1
            bin_state = 0
            for __ in range(mutation_num): 
                probs = np.exp(self.log_theta[np.ix_(S, evs)].sum(axis=1) + self.log_theta[S, S]) / (1 - restr_diag[bin_state + (1 << np.array(S))])
                i = np.random.choice(
                    np.arange(len(S)),
                    p=probs/probs.sum())
                evs.append(S.pop(i))
                p *= (probs[i] / probs.sum())
                bin_state += (1 << evs[-1])
            return evs, p
        
        samples = list()    
        
        n = 0
        evs, last_p = proposal()
        samples.append(evs)

        while n < int((1 + burn_in) * n_samples):
            evs, p = proposal()
            
            if np.random.random() <= min(1, p/last_p):
                samples.append(evs)
                last_p = p
                n += 1
        
        return samples[-n_samples:]