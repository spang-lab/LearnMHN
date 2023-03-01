"""
This submodule contains a class to represent Mutual Hazard Networks
"""
# author(s): Y. Linda Hu, Stefan Vocht

from __future__ import annotations

import numpy as np
import pandas as pd
import json


class MHN:
    """
    This class represents the Mutual Hazard Network
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

    def get_random_sample(self, sample_num: int, as_dataframe: bool = False) -> np.ndarray | pd.DataFrame:
        """
        Returns artificial data sampled from this MHN. Random values are generated with numpy, use np.random.seed()
        to make results reproducible.

        :param sample_num: number of samples in the generated data
        :param as_dataframe: if True, the data is returned as a pandas DataFrame
        """
        n = self.log_theta.shape[0]
        exp_theta = np.exp(self.log_theta)
        art_data = np.zeros((sample_num, n), dtype=np.int32)
        for sample_index in range(sample_num):
            current_sample = []
            observation_time = np.random.exponential(1)
            current_time = 0
            while len(current_sample) < n:
                rates_from_current_state = []
                possible_gene_mutations = []
                for j in range(n):
                    if j not in current_sample:
                        possible_gene_mutations.append(j)
                        rate = exp_theta[j, j]
                        for mutated_gene in current_sample:
                            rate *= exp_theta[j, mutated_gene]
                        rates_from_current_state.append(rate)

                q_ii = sum(rates_from_current_state)
                passed_time = np.random.exponential(1/q_ii)
                current_time += passed_time
                if current_time > observation_time:
                    break
                random_crit = np.random.random(1)[0] * q_ii
                accumulated_rate = 0
                for rate, gene in zip(rates_from_current_state, possible_gene_mutations):
                    accumulated_rate += rate
                    if random_crit <= accumulated_rate:
                        current_sample.append(gene)
                        art_data[sample_index, gene] = 1
                        break

        if as_dataframe:
            df = pd.DataFrame(art_data)
            if self.events is not None:
                df.columns = self.events
            return df
        else:
            return art_data

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
        :param events: list of strings containing the names of the events considered by the MHN
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
