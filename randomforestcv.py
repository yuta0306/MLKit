try:
    from typing import Any, AnyStr, Dict, Final, Generic, Iterable, Optional, Union, List, TypeVar
except ImportError:
    from typing import Any, AnyStr, Dict, Generic, Iterable, Optional, Union, List, TypeVar
from dataclasses import dataclass, field, InitVar
from itertools import product
import re
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

T = TypeVar('T')

@dataclass
class RandomForestRegressorGS(Generic[T]):
    n_estimators: List[Union[AnyStr, int]] = field(default_factory=lambda: ['warn'])
    criterion: List[AnyStr] = field(default_factory=lambda: ["mse"])
    max_depth: List[Optional[int]] = field(default_factory=lambda: [None])
    min_samples_split: List[int] = field(default_factory=lambda: [2])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1])
    min_weight_fraction_leaf: List[int] = field(default_factory=lambda: [0.])
    max_features: List[Union[AnyStr, int]] = field(default_factory=lambda: ["auto"])
    max_leaf_nodes: List[Optional[int]] = field(default_factory=lambda: [None])
    min_impurity_decrease: List[float] = field(default_factory=lambda: [0.])
    min_impurity_split: List[Optional[int]] = field(default_factory=lambda: [None])
    bootstrap: List[bool] = field(default_factory=lambda: [True])
    oob_score: List[bool] = field(default_factory=lambda: [False])
    n_jobs: List[Optional[int]] = field(default_factory=lambda: [None])
    random_state: List[Optional[int]] = field(default_factory=lambda: [None])
    verbose: List[int] = field(default_factory=lambda: [0])
    warm_start: List[bool] = field(default_factory=lambda: [False])
    _bestmodel: Any = None
    _bestscore: Optional[float] = None
    _best_params: Optional[List] = None


    def fit(self, X, y) -> T:
        kwargs_instance: Dict[Iterable] = vars(self)
        kwargs: Dict[Iterable] = kwargs_instance.copy()
        for k, v in kwargs_instance.items():
            if re.match('\_', k):
                kwargs.pop(k)
            else:
                if not isinstance(v, Iterable):
                    kwargs[k] = [v]

        p = product(*(kwargs.values()))
        for comb in tqdm(p):
            params = {k: v for k, v in zip(kwargs.keys(), comb)}
            rfr: RandomForestRegressor = RandomForestRegressor(**params)
            rfr.fit(X, y)

            score = rfr.score(X, y)
            if self._bestscore is None:
                self._bestscore = score
                self._bestmodel = rfr
            else:
                if score > self._bestscore:
                    self._bestscore = score
                    self._bestmodel = rfr

        return self

    def predict(self, X) -> Iterable:
        pred = self._bestmodel.predict(X)

        return pred

    def score(self, X, y) -> float:
        score = self._bestmodel.score(X, y)

        return score

    @property
    def bestmodel(self):
        return self._bestmodel
    
    @property
    def bestscore(self):
        return self._bestscore
