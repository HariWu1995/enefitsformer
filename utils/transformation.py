"""
Customized Transformer(s) for SkLearn
    https://www.andrewvillazon.com/custom-scikit-learn-transformers/
"""
import itertools
import numpy as np
import pandas as pd
import polars as pl

from sklearn.experimental import enable_iterative_imputer
from sklearn.utils._encode import _unique, _encode
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, _check_feature_names_in, column_or_1d, _num_samples

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, \
                                   OneHotEncoder, LabelEncoder, OrdinalEncoder, MultiLabelBinarizer


class OrdinalCumulator(TransformerMixin):

    def get_feature_names_out(self, input_features=None):
        # input_features = _check_feature_names_in(self, input_features)
        return self.encoder.classes_
    
    def __init__(self, *args, **kwargs):
        # Use MultiLabelBinarizer to handle UNKNOWN values
        self.encoder = MultiLabelBinarizer(*args, **kwargs)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
        
    def fit(self, X, y=0):
        self.encoder.fit(X)
        return self
    
    def transform(self, X, y=None):
        # 1-hot encoding
        X = self.encoder.transform(X)
        
        # Cumulatation --> better for ordinal classification
        X = pd.DataFrame(X).T.replace({0: np.nan}).fillna(method='bfill').fillna(0).T
        return X.values
    
    def inverse_transform(self, X, y=None):
        # Validate
        check_is_fitted(self.encoder)
        if X.shape[1] != len(self.encoder.classes_):
            raise ValueError(
                f"Expected indicator for {len(self.encoder.classes_)} classes, but got {X.shape[1]}")

        # Inverse-transform
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_ = [np.where(x==1)[0] for x in X]
        X_ = [x[-1] if len(x) > 0 else -1 for x in X_]
        X_ = np.array(X_).reshape(-1,1)
            
        return X_


class MultiLabelEncoder(MultiLabelBinarizer):

    def get_feature_names_out(self, input_features=None):
        # input_features = _check_feature_names_in(self, input_features)
        return self.classes_
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_transform(self, y, X=None):
        self.fit(y)
        return self.transform(y)
        
    def fit(self, y, X=None):
        super().fit(y)
        return self
    
    def transform(self, y, X=None):
        return super().transform(y)


class UniLabelEncoder(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):

    def get_feature_names_out(self, input_features=None):
        input_features = _check_feature_names_in(self, input_features)
        return input_features

    def fit(self, y, X=None):
        if isinstance(y, pd.Series):
            y = y.values
        classes = sorted(list(np.unique(y)))
            
        dtype = int if all(isinstance(c, int) for c in classes) else object
        self.classes_ = np.empty(len(classes), dtype=dtype)
        self.classes_[:] = classes
        return self

    def fit_transform(self, y, X=None):
        self.fit(y)
        y_out = self.transform(y)
        return y_out

    def _transform(self, y, X=None):
        check_is_fitted(self)
        y = column_or_1d(y, dtype=self.classes_.dtype, warn=True)
        
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        return _encode(y, uniques=self.classes_)
    
    def transform(self, y, X=None):
        if not isinstance(y, pd.Series):
            if isinstance(y, pl.Series):
                y = y.to_pandas()
            elif isinstance(y, (np.ndarray, list, tuple)):
                y = pd.Series(y)
            else:
                raise ValueError(f'{y.__class__} is not supported!')
        else:
            y = y

        y_out = pd.Series([np.nan] * len(y))

        # Detect outliers
        is_outlier = ~y.isin(self.classes_)

        # Handle outliers
        y_out.loc[is_outlier] = -1

        # Process inliers, as normal
        y_out.loc[~is_outlier] = self._transform(y.loc[~is_outlier])

        return y_out.values.reshape(-1,1)
        
    def inverse_transform(self, y, X=None):
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        
        # inverse transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if len(diff):
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        # return self.classes_[y]
        return [i if i in self.classes_ else np.nan for i in y]
        

class BooleanPolarizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, pos_values=None, neg_values=None, columns=None):
        self.neg_values = neg_values if neg_values else [False, 'false', 'F', 0]
        self.pos_values = pos_values if pos_values else [ True,  'true', 'T', 1]
        self.columns = columns

    def get_feature_names_out(self, input_features=None):
        input_features = _check_feature_names_in(self, input_features)
        return input_features
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if isinstance(X, (np.ndarray, list, tuple)):
            X = pd.DataFrame(X)

        if self.columns:
            cols_to_transform = self.columns
        else:
            cols_to_transform = list(X.columns)

        Yp = X[cols_to_transform].isin(self.pos_values).astype(int)
        Yn = X[cols_to_transform].isin(self.neg_values).astype(int)
        Y = Yp - Yn  # Positives = 1, Negatives = -1, Outlier = 0
        return Y



