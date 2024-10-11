from typing import Dict, Union

import pandas as pd
import polars as pl

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class TsDataLoaderSimple(Dataset):
    """
    Time-series Dataset using Single-DataTable
    """
    def __init__(self, dataset: pl.DataFrame,
                       groupby_columns = [],
                        ignore_columns = [],
                         count_column  = 'COUNT',
                         forecast_window: int = 1,
                         lookback_window: int = 1, 
                              batch_size: int = 1, 
                               time_skip: int = 1,
                             is_shuffled: bool = False,):

        self.time_skip = time_skip
        self.batch_size = batch_size
        self.is_shuffled = is_shuffled
        self.count_column = count_column
        self.groupby_columns = groupby_columns
        self.forecast_window = forecast_window
        self.lookback_window = lookback_window
        self.temporal_window = lookback_window + forecast_window
        self.dataset = dataset
        if len(ignore_columns) > 0:
            self.dataset = self.dataset.drop(columns=ignore_columns)
        self.build_indexer()

    def build_indexer(self):
        index_count = (pl.col(self.count_column) - self.temporal_window).alias('INDEX_COUNT')
        index_start = (pl.col('INDEX_COUNT').shift(1, fill_value=-1) + 1).cum_sum().alias('INDEX_START')
        index_end_ = (pl.col('INDEX_COUNT') + pl.col('INDEX_START')).alias('INDEX_END')
        index_local = pl.col('INDEX_COUNT').apply(lambda x: list(range(0, x+1))).alias('INDEX_LOCAL')
        index_columns = ['INDEX_LOCAL']
        self.indexer = self.dataset.select(self.groupby_columns+[self.count_column])\
                        .with_columns(index_count).with_columns(index_start)\
                         .with_columns(index_end_).with_columns(index_local)\
                                   .select(self.groupby_columns+index_columns)\
                                               .explode(columns=index_columns)

        if self.time_skip > 1:
            self.indexer = self.indexer.filter(pl.col('INDEX_LOCAL').mod(self.time_skip) == 0)
        self.indexer = self.indexer.with_columns(INDEX_GLOBAL = pl.lit(list(range(len(self.indexer)))))

        # Check duplicated indices
        # print(self.indexer.filter(pl.col("INDEX_GLOBAL").is_duplicated()))

    def shuffle(self, random_seed=None):
        # self.indexer = self.indexer.sample(fraction=1.0, shuffle=True)
        self.indexer = self.indexer.with_columns(
                    pl.col('INDEX_GLOBAL').shuffle(seed=random_seed))

    def __len__(self):
        return (len(self.indexer) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        """                      (1)                     (2)
        Direction: INDEX_GLOBAL ---->> GROUP_BY COLUMNS ---->> INDEX_LOCAL --->> (B, L)
        """
        if (index < 0) or (index >= self.__len__()):
            raise IndexError(f"index {index} is out of range [0, {self.__len__()-1}] !")
        if (index == 0) and self.is_shuffled:
            self.shuffle()

        # (1) Convert global indices to per-group local indices
        index_start = self.batch_size * index
        index_end = min(self.batch_size * (index + 1), len(self.indexer))
        index_map = self.indexer.filter(
            pl.col('INDEX_GLOBAL').is_between(index_start, index_end, closed='left')
        ).group_by(self.groupby_columns).agg([pl.col('INDEX_LOCAL').flatten()])
        if len(index_map) == 0:
            print(f'[WARNING!] No sample is indexed between [{index_start}, {index_end}], ' + \
                  f'batch size = {index_end - index_start} for index = {index}')
            print(index_map)

        # (2) Gather per-group per-slice batch-data, then, stack
        drop_columns = self.groupby_columns + [self.count_column]
        data_lookback = {col: [] for col in self.dataset.columns if col not in drop_columns}
        data_forecast = {col: [] for col in self.dataset.columns if col not in drop_columns}

        for X in index_map.iter_rows(named=True):
            X_idx = X.pop('INDEX_LOCAL')
            X_cond = [pl.col(k) == v for k, v in X.items()]
            X_ts = self.dataset.filter(*X_cond).drop(columns=drop_columns)
            for idx_0 in X_idx:
                idx_1 = idx_0 + self.lookback_window
                idx_2 = idx_1 + self.forecast_window
                for col in X_ts.columns:
                    col_data = X_ts[col].to_list()[0]
                    col_X = col_data[idx_0:idx_1]
                    col_Y = col_data[idx_1:idx_2]
                    data_lookback[col].append(col_X)
                    data_forecast[col].append(col_Y)
        try:
            data_lookback = {k: torch.tensor(v) for k, v in data_lookback.items()}
            data_forecast = {k: torch.tensor(v) for k, v in data_forecast.items()}
        except:
            return data_lookback, data_forecast
            
        if any([(v.shape[0] == 0) for v in data_lookback.values()]):
            return None, None
        return data_lookback, data_forecast


class TsDataLoaderMixed(TsDataLoaderSimple):
    """
    Time-series Dataset using Multiple-DataTables
    """
    def __init__(self, datasets: Dict[str, pl.DataFrame], **kwargs):
        kwargs['dataset'] = datasets.pop('main')
        super().__init__(**kwargs)
        self.datasets_aux = datasets

    def __getitem__(self, index):
        pass


class TsDataLoaderForClassifier(Dataset):

    def __init__(self, data: torch.Tensor, label: torch.Tensor, lookback_window: int = 1, device=None):
        self.data = data
        self.label = label
        if device:
            self.data.to(device)
            self.label.to(device)
        self.lookback_window = lookback_window

    def __len__(self):
        return self.data.shape[1] - (self.lookback_window-1)

    def __getitem__(self, index):
        X = self.data[:, index:index+self.lookback_window]
        y = self.label[:,      index+self.lookback_window]
        return X, y



