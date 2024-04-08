import numpy as np
from typing import Union, Dict, Literal, TypeVar
from .relations import PositivePreference, NegativePreference, Incomparible, Indifference

RelationType = Union[PositivePreference, NegativePreference, Incomparible, Indifference]
ScoreType = Dict[RelationType, Dict[RelationType, int]]
RankingType = Literal["partial", "complete"]
StochasticType = Dict[RelationType, np.ndarray]

QuadraticArrayType = np.ndarray

