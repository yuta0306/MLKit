import sklearn
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

from typing import Any, List, Union

__all__ = [
    'measure_importances'
]

def measure_importances(model: Any, X: List or np.ndarray or pd.DataFrame or pd.Series,
                        y: List or np.ndarray or pd.DataFrame or pd.Series, n_split: int = 5,
                        shuffle: bool = False, random_state: Union[int] = None) -> np.ndarray:

    if 'feature_importances_' not in dir(model):
        raise ValueError('model has no attribute `feature_importances_`')

    kf: KFold = KFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)
    kf.get_n_splits(X)

    importances: Union[np.ndarray] = None

    for train_idx, _ in kf.split(X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_train: pd.DataFrame or pd.Series = X.iloc[train_idx]
        else:
            X_train = X[train_idx]

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_train: pd.DataFrame or pd.Series = y.iloc[train_idx]
        else:
            y_train = y[train_idx]

        model.fit(X_train, y_train)

        if importances is None:
            importances = model.feature_importances_
        else:
            importances = importances + model.feature_importances_

    importances = importances / n_split

    return importances