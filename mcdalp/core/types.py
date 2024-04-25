import numpy as np
from typing import Union, Dict, Literal, TypeVar
from .relations import PositivePreference, NegativePreference, Incomparible, Indifference
from .const import RankingMode

RelationType = Union[PositivePreference, NegativePreference, Incomparible, Indifference]
ScoreType = Dict[RelationType, Dict[RelationType, int]]
RankingModeType = Literal[RankingMode.PARTIAL, RankingMode.COMPLETE]
StochasticType = Dict[RelationType, np.ndarray]

QuadraticArrayType = np.ndarray

