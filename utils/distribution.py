import pandas as pd
import numpy as np
import scipy as sp


def scale(sr: pd.Series or np.ndarray, mode: str = 'min_max'):
    if mode == 'min_max':
        sr = (sr - sr.min()) / (sr.max() - sr.min())
    elif mode == 'standard':
        sr = (sr - sr.mean()) / sr.std()
    return sr


def uniformize(x, nbins: int = 100):
    step = 1. / nbins

    mapping = pd.DataFrame(data={'cdf': np.arange(0, 1+step, step)})
    mapping['value'] = mapping['cdf'].apply(lambda p: np.quantile(x, p))
    mapping.drop_duplicates(subset=['value'], inplace=True)

    func = sp.interpolate.interp1d(mapping['value'], 
                                   mapping['cdf'], kind='linear', fill_value="extrapolate")
    return func(x)


