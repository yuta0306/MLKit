try:
    from typing import Any, AnyStr, Dict, Final, Generic, Iterable, Optional, Union, List, TypeVar
except ImportError:
    from typing import Any, AnyStr, Dict, Generic, Iterable, Optional, Union, List, TypeVar
from dataclasses import dataclass, field, InitVar
from itertools import product

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

T = TypeVar('T')

@dataclass
class RandomForestCV(Generic[T]):
    n_estimators: InitVar[List[Union[AnyStr, int]]] = ['warn']
    criterion: InitVar[List[AnyStr]] = ["mse"]
    max_depth: InitVar[List[Optional[int]]] = [None]
    min_samples_split: InitVar[List[int]] = [2]
    min_samples_leaf: InitVar[List[int]] = [1]
    min_weight_fraction_leaf: InitVar[List[int]] = [0.]
    max_features: InitVar[List[Union[AnyStr, int]]] = ["auto"]
    max_leaf_nodes: InitVar[List[Optional[int]]] = [None]
    min_impurity_decrease: InitVar[List[float]] = [0.]
    min_impurity_split: InitVar[List[Optional[int]]] = [None]
    bootstrap: InitVar[List[bool]] = [True]
    oob_score: InitVar[List[bool]] = [False]
    n_jobs: InitVar[List[Optional[int]]] = [None]
    random_state: InitVar[List[Optional[int]]] = [None]
    verbose: InitVar[List[int]] = [0]
    warm_start: InitVar[List[bool]] = [False]
    bestmodel: Any = None
    bestscore: Optional[float] = None


    def fit(self, X, y, X_eval, y_eval) -> T:
        instant_kwargs: Dict[Any]= vars(self)
        kwargs: Dict[Iterable] = instant_kwargs.copy()
        basemodel: Optional[RandomForestRegressor] = None
        for k, v in instant_kwargs.items():
            if not isinstance(v, Iterable):
                kwargs.pop(k)

        COMB: List[Dict[Any]] = product(*kwargs)

        for values in COMB:
            params: Dict[Any] = {k: v for k, v in zip(kwargs.keys(), values)}

            basemodel = RandomForestRegressor(**params).fit(X, y)
            
            score = basemodel.score(X_eval, y_eval)

            if self.bestscore is None:
                self.bestscore = score
                self.bestmodel = basemodel
            else:
                if score > self.bestscore:
                    self.bestscore = score
                    self.bestmodel = basemodel

        return self

    def predict(self, X) -> Iterable:
        pred = self.bestmodel.predict(X)

        return pred

    def score(self, X, y) -> float:
        score = self.bestmodel.score(X, y)

        return score