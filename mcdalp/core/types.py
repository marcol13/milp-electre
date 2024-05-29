import numpy as np
from typing import Union, Dict, Literal, Tuple
from .relations import PositivePreference, NegativePreference, Incomparible, Indifference
from .const import RankingMode, INPUT_TYPES

RelationType = Union[PositivePreference, NegativePreference, Incomparible, Indifference]
ScoreType = Dict[RelationType, Dict[RelationType, int]]
RankingModeType = Literal[RankingMode.PARTIAL, RankingMode.COMPLETE]
StochasticType = Dict[RelationType, np.ndarray]
InputType = Tuple[INPUT_TYPES]

QuadraticArrayType = np.ndarray

