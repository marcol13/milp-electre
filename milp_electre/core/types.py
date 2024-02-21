from typing import Union, Dict
from .relations import PositivePreference, NegativePreference, Incomparible, Indifference

RelationType = Union[PositivePreference, NegativePreference, Incomparible, Indifference]
ScoreType = Dict[RelationType, Dict[RelationType, int]]