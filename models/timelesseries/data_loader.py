from typing import Dict, List, Union

import pandas as pd
import polars as pl

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle


class DataLoaderSimple(Dataset):

    def __init__(self, dataset: pl.DataFrame or pd.DataFrame,
                       targets: Union[str, List, Dict], ignore_columns = [],
                    batch_size: int = 1, is_shuffled: bool = False,):

        if isinstance(targets, str):
            self.targets = [targets]
        elif isinstance(targets, dict):
            self.targets = {
                k: [v] if isinstance(v, str) else v
                for k, v in targets.items()
            }
        elif not isinstance(targets, (list, tuple)):
            raise ValueError(f'targets of type {targets.__class__} is not supported!')

        self.batch_size = batch_size
        self.is_shuffled = is_shuffled

        if isinstance(dataset, pd.DataFrame):
            self.dataset = pl.from_pandas(dataset)
        elif isinstance(dataset, pl.DataFrame):
            self.dataset = dataset
        else:
            raise TypeError('dataset must be dataframe of either Polars or Pandas, ' + \
                            f'while input type is {dataset.__class__}')

        if len(ignore_columns) > 0:
            self.dataset = self.dataset.drop(columns=ignore_columns)
        self.build_indexer()

    def build_indexer(self):
        self.dataset = self.dataset.with_columns(index=pl.lit(list(range(len(self.dataset)))))

    def shuffle(self, random_seed=None):
        self.dataset = self.dataset.with_columns(
                    pl.col('index').shuffle(seed=random_seed))

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):

        idx_start = self.batch_size * index
        idx_end = self.batch_size * (index+1)
        batch_data = self.dataset.filter((pl.col('index') >= idx_start) & \
                                         (pl.col('index') < idx_end))\
                                  .drop(columns=['index']).to_pandas()

        if isinstance(self.targets, (list, tuple)):
            Y = torch.tensor(batch_data[self.targets].values)
            batch_data = batch_data.drop(columns=self.targets)
        elif isinstance(self.targets, dict):
            Y = dict()
            for k, v in self.targets.items():
                Y[k] = torch.tensor(batch_data[v].values)
                batch_data = batch_data.drop(columns=v)

        X = {
            col: torch.tensor(batch_data[[col]].values) for col in batch_data.columns
        }
        return X, Y


class DataLoaderMixed(DataLoaderSimple):
    """
    Time-series Dataset using Multiple-DataTables
    """
    def __init__(self, datasets: Dict[str, pl.DataFrame], **kwargs):
        kwargs['dataset'] = datasets.pop('main')
        super().__init__(**kwargs)
        self.datasets_aux = datasets

    def __getitem__(self, index):
        pass



