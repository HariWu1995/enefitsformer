from tqdm import tqdm

import numpy as np
import pandas as pd
import polars as pl

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

from .transformation import *


def build_pipeline(**kwargs):

    preprocessor = dict()
    
    # Boolean features: BooleanPolarizer (Positve = 1, Negative = -1, Outlier = 0)
    preprocessor['boolean'] = Pipeline(steps = [
        ("polarize", BooleanPolarizer()),
    ])

    # Numerical features: Imputation (Mean) --> Scaler (Standard)
    preprocessor['numerical'] = Pipeline(steps = [
        # ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scale", StandardScaler()),
    ])

    # Ordinal features: OrdinalEncoder --> OrdinalCumulator
    preprocessor['ordinal'] = Pipeline(steps = [
        ("encode", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, 
                                                              encoded_missing_value=-1, )),
        ("cumulate", OrdinalCumulator()),
    ])

    # Categorical features: Label-Encoder with Imputation (Outlier = -1) 
    # NOTE: this transformer has issue when dealing with `object`
    preprocessor['categorical'] = Pipeline(steps = [
        ("encode", UniLabelEncoder()),
    ])

    # Multi-label features: MultiLabelBinarizer (Outlier = [0] * N)
    preprocessor['mulilabel'] = Pipeline(steps = [
        ("encode", MultiLabelEncoder(sparse_output=False)),
    ])

    # Build pipeline
    features_pipeline = []
    for dtype in ['numerical','boolean','categorical','ordinal','multilabel']: 
        for col in kwargs.get(f"{dtype}_columns", []):
            features_pipeline.append(
                        (f'{dtype}_{col}', 
               preprocessor[dtype], [col] if dtype in ['numerical','boolean','ordinal'] else col)
            )
    passthrough_columns = kwargs.get('passthrough_columns', [])
    if len(passthrough_columns) > 0:
        features_pipeline.append(
            ("original_", "passthrough", passthrough_columns)
        )
    
    features_pipeline = ColumnTransformer(features_pipeline, remainder='drop', verbose=True)
    return features_pipeline


def preprocess_features(dataset_df: pd.DataFrame or pl.DataFrame,
                        categorical_features: list=[],
                          numerical_features: list=[],
                            boolean_features: list=[],
                         multilabel_features: list=[],
                       classification_labels: list=[],
                           regression_labels: list=[],) -> (pd.DataFrame, dict):
    
    if isinstance(dataset_df, pl.DataFrame):
        dataset_df = dataset_df.to_pandas()
    
    Schema = dict()
    print("Preprocessing features ...")

    # Schema for numerical features
    print("\t Preprocessing numerical features ...")
    for feat in tqdm(numerical_features):
        value_min = dataset_df[feat].values.min()
        scaler = StandardScaler() if value_min < 0 else MinMaxScaler()
        dataset_df[feat] = scaler.fit_transform(dataset_df[feat].values.reshape(-1, 1))
        Schema[feat] = {'type': "feature_numerical", 'scaler': scaler,}

    # Schema for categorical features
    print("\t Preprocessing categorical features ...")
    for feat in tqdm(categorical_features):
        encoder = LabelEncoder()
        dataset_df[feat] = encoder.fit_transform(dataset_df[feat].values) + 1
        Schema[feat] = {'type': "feature_categorical", 'encoder': encoder,}

    # Schema for multi-label features
    print("\t Preprocessing multilabel features ...")
    for feat in tqdm(multilabel_features):
        encoder = MultiLabelBinarizer()
        dataset_df[feat] = np.clip(encoder.fit_transform(dataset_df[feat].values), 0.1, 0.9).tolist() # label smoothing
        Schema[feat] = {'type': "feature_multilabel", 'encoder': encoder,}

    # Schema for labels for classification
    print("\t Preprocessing classification labels ...")
    for feat in tqdm(classification_labels):
        encoder = LabelEncoder()
        dataset_df[feat] = encoder.fit_transform(dataset_df[feat].values) + 1
        Schema[feat] = {'type': "label_classification", 'encoder': encoder,}

    # Schema for labels for regression
    print("\t Preprocessing classification labels ...")
    for feat in tqdm(regression_labels):
        scaler = StandardScaler()
        dataset_df[feat] = scaler.fit_transform(dataset_df[feat].values.reshape(-1, 1))
        Schema[feat] = {'type': "label_regression", 'scaler': scaler,}
        
    return dataset_df, Schema
    