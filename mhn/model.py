# by Y. Linda Hu

import numpy as np
import pandas as pd
import json


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
