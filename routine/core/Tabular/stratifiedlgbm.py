import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error as mse
import lightgbm as lgb

class StratifiedLGBM:
    def __init__(self, n_splits: int=5, shuffle: bool=True, random_state: int=2021):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        self.models = None
        self.oof = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, params: dict, *args, **kwargs):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                                random_state=self.random_state)
        self.oof = np.zeros_like(y)
        self.models = list()
        for tr_idx, val_idx in skf.split(X, y.astype(int)):
            X_train, X_val = X.iloc[tr_idx, :], X.iloc[val_idx, :]
            y_train, y_val = y[tr_idx], y[val_idx]
            
            trainset = lgb.Dataset(X_train, y_train)
            validset = lgb.Dataset(X_val, y_val, reference=trainset)
            
            model = lgb.train(params, trainset, valid_sets=[trainset, validset],
                             valid_names=['Train', 'Valid'], **kwargs)
            
            preds = model.predict(X_val)
            self.oof[val_idx] = preds
            self.models.append(model)
        print(f'CV >>>>> {mse(y, self.oof)}')
        
        return self
    
    def predict(self, X):
        assert self.models is not None  # すでに学習済みかテストする
        
        preds = [
            model.predict(X)
                for model in self.models
        ]
        preds = np.mean(preds, axis=0)
        
        return preds
        
    def plot_importance(self, importance_type='gain', max_range=50):
        feature_importance_df = pd.DataFrame()
        for i, model in enumerate(self.models):
            df = pd.DataFrame()
            df['feature_importance'] = model.feature_importance(importance_type=importance_type)
            df['column'] = model.feature_name()
            df['fold'] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, df],
                                              axis=0, ignore_index=True)
            
        order = feature_importance_df.groupby('column').sum()[['feature_importance']]\
                    .sort_values('feature_importance', ascending=False).index[:max_range]
        
        fig, ax = plt.subplots(figsize=(max(6, len(order) * .4), 7))
        sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance',
                        order=order, ax=ax, palette='viridis')
        ax.tick_params(axis='x', rotation=45)
        ax.grid()
        fig.tight_layout()
        