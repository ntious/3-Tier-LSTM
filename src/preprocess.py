from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

def drop_unnecessary(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=cols, errors="ignore")

def encode_target(df: pd.DataFrame, target_col: str, pos_label: str = "M", neg_label: str = "B") -> pd.DataFrame:
    df = df.copy()
    if df[target_col].dtype == object:
        mapping = {pos_label: 1, neg_label: 0}
        df[target_col] = df[target_col].map(mapping)
    return df

def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    stratify: bool = True,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)

def feature_selection_dt(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: Optional[int] = 5,
    random_state: int = 42,
    top_k: Optional[int] = None
) -> Tuple[List[str], np.ndarray]:
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)
    importances = dt.feature_importances_
    feature_order = np.argsort(importances)[::-1]
    feature_names_sorted = X_train.columns[feature_order]
    if top_k is not None:
        feature_names_sorted = feature_names_sorted[:top_k]
    return list(feature_names_sorted), importances

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, object]:
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unknown scaling method: choose 'standard' or 'minmax'")
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

def apply_smote(
    X_train: np.ndarray,
    y_train: pd.Series,
    k_neighbors: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    sm = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res
