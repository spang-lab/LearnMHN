# by Y. Linda Hu

import numpy as np
import pandas as pd
class bits_fixed_n:

    def __init__(self, n, k):
        self.v = int("1"*n, 2)
        self.stop_no = int("1"*n + "0"*(k-n), 2)
        self.stop=False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration
        if self.v == self.stop_no:
            self.stop=True
        t = (self.v | (self.v - 1)) + 1
        w = t | ((((t & -t)) // (self.v & (-self.v)) >> 1) - 1)
        self.v, w = w, self.v
        return w


class MHN:
    """
    This class represents the Mutual Hazard Network
    """

    def __init__(self, log_theta: np.ndarray, events: list[str] = None):

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
