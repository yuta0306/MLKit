import pandas as pd


class GroupingAggregation:
    def __init__(self, group_key: str, group_values: list or tuple, agg_methods: list or tuple):
        """
        GroupingAggregation:
            The class for aggregation with group
        """
        self.group_key = group_key
        self.group_values = group_values
        
        excludes = ['val-mean', 'z-score']
        self.ex_trans_methods = [method for method in agg_methods if method in excludes]
        self.agg_methods = [method for method in agg_methods if not method in excludes]
        
    def fit(self, input_: pd.DataFrame, y=None):
        new_dfs = []
        for agg_method in self.agg_methods:
            for col in self.group_values:
                if callable(agg_method):
                    agg_method_name = agg_method.__name__
                else:
                    agg_method_name = agg_method
                    
                new_col = f'agg_{agg_method_name}_{col}_groupby_{self.group_key}'
                df_agg = (input_[[col] + [self.group_key]].groupby(self.group_key)[[col]].agg(agg_method))
                df_agg.columns = [new_col]
                new_dfs.append(df_agg)
        self.df = pd.concat(new_dfs, axis=1).reset_index()
        
        return self
    
    def transform(self, input_: pd.DataFrame):
        assert hasattr(self, 'df')
        output = pd.merge(input_[[self.group_key]], self.df, on=self.group_key, how='left')
        if len(self.ex_trans_methods) > 0:
            output = self.ex_transform(input_, output)
        output = output.drop(self.group_key, axis=1)
        
        return output
    
    def fit_transform(self, input_: pd.DataFrame, y=None):
        self.fit(input_, y=y)
        return self.transform(input_)
            
    def ex_transform(self, input_: pd.DataFrame, output: pd.DataFrame):
        if "val-mean" in self.ex_trans_methods:
            output[self._get_col("val-mean")] = input_[self.group_values].values - output[self._get_col("mean")].values
        if "z-score" in self.ex_trans_methods:
            output[self._get_col("z-score")] = (input_[self.group_values].values - output[self._get_col("mean")].values) \
                                            / (output[self._get_col("std")].values + 1e-3)
        return output
    
    def _get_col(self, method: str):
        return [f"agg_{method}_{group_val}_groupby_{self.group_key}" for group_val in self.group_values]