import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class TargetEncoding:
    def __init__(self, categories, target):
        assert isinstance(categories, (str, list, tuple, np.ndarray))
        assert isinstance(target, (str, pd.Series, np.ndarray))
        
        if isinstance(categories, (str)):
            self.categories = [categories]
        elif isinstance(categories, (list, tuple, np.ndarray)):
            self.categories = categories
        
        self.target = target
        self.encoders = None
        self.fold = None
        
    def fit(self, X: pd.DataFrame, y=None, fold=None, n_splits=10, shuffle=True, random_state=2021):
        assert isinstance(X, pd.DataFrame)
        
        self.target = y if y is not None else self.target
        if isinstance(self.target, str):
            target = X.loc[:, self.target]
        else:
            target = self.target

        fold = fold if fold is not None else KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.fold = fold
        
        self.encoders = list()
        for tr_idx, va_idx in fold.split(X):
            fold_encoders = list()
            X_train = X.loc[tr_idx, self.categories]
            y_train = target[tr_idx]
            
            df = pd.concat([X_train, y_train], axis=1)
            fold_encoders = [
                df.groupby(cat).mean()
                    for cat in self.categories
            ]
            self.encoders.append(fold_encoders)
            
        return self
    
    def transform(self, X, cv=False):
        assert self.encoders is not None
        
        if cv:
            output = np.array([
                np.repeat(np.nan, X.shape[0]) for _ in range(len(self.categories))
            ])
            for encoder, (_, va_idx) in zip(self.encoders, self.fold.split(X)):
                X_val = X.loc[va_idx, self.categories]
                
                for i, (cat, cat_enc) in enumerate(zip(self.categories, encoder)):
                    tmp = pd.merge(X_val.loc[:, cat], cat_enc, left_on=cat, right_index=True, how='left').drop(cat, axis=1)
                    output[i][va_idx] = tmp.values.flatten()
                    
            output = pd.DataFrame(zip(*output), columns=[f'TE_{cat}' for cat in self.categories])
                
        else:
            encoders = [
                pd.concat(encoder, axis=1) for encoder in zip(*self.encoders)
            ]
            encoders = [
                encoder.mean(axis=1).rename(f'TE_{cat}') for cat, encoder in zip(self.categories, encoders)
            ]
            
            decorders = [
                pd.merge(X.loc[:, cat], enc, left_on=cat, right_index=True)
                    for cat, enc in zip(self.categories, encoders)
            ]
            
            output = pd.concat(decorders, axis=1).drop(self.categories, axis=1)
        
        return output
            
    def fit_transform(self, X: pd.DataFrame, y=None, fold=None, n_splits=10, shuffle=True, random_state=2021, cv=False):
        self.fit(X, y, fold, n_splits, shuffle, random_state)
        output = self.transform(X, cv)
        
        return output