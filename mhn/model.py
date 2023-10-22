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


class MHN:
    """
    This class represents a Mutual Hazard Network.
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
        :param as_dataframe: if True, the data is returned as a pandas DataFrame, else numpy matrix

        :returns: array or DataFrame with samples as rows and events as columns
        """
        art_data = Likelihood.sample_artificial_data(self.log_theta, sample_num)
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
        p_th = state_space_restriction.compute_restricted_inverse(self.log_theta, state, p0, False)
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

    def __str__(self):
        if isinstance(self.meta, dict):
            meta_data_string = '\n'.join([f'{key}:\n{value}\n' for key, value in self.meta.items()])
        else:
            meta_data_string = "None"
        return f"EVENTS: \n{self.events}\n\n" \
               f"THETA IN LOG FORMAT: \n {self.log_theta}\n\n" \
               f"ADDITIONAL METADATA: \n\n{meta_data_string}"


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
        return self.get_equivalent_vanilla_mhn().sample_artificial_data(sample_num, as_dataframe)

    def compute_marginal_likelihood(self, state: np.ndarray) -> float:
        """
        Computes the likelihood of observing a given state, where we consider the observation time to be an
        exponential random variable with mean 1.

        :param state: a 1d numpy array (dtype=np.int32) containing 0s and 1s, where each entry represents an event being present (1) or not (0)

        :returns: the likelihood of observing the given state according to this MHN
        """
        return self.get_equivalent_vanilla_mhn().compute_marginal_likelihood(state)

    def get_equivalent_vanilla_mhn(self) -> MHN:
        """
        This method returns a vanilla MHN object that represents the same distribution as this OmegaMHN object.

        :returns: vanilla MHN object representing the same distribution as this OmegaMHN object
        """
        n = self.log_theta.shape[1]
        # subtract observation rates from each element in each column
        equivalent_vanilla_mhn = self.log_theta[:-1] - self.log_theta[-1]
        # undo changes to the diagonal
        equivalent_vanilla_mhn[range(n), range(n)] += self.log_theta[-1]
        return MHN(equivalent_vanilla_mhn, self.events, self.meta)

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
            with open(f"{filename}_meta.json", "x") as file:
                json.dump(self.meta, file, indent=4)
