from typing import NamedTuple
from mcdalp.core.types import RankingMode


class ThresholdType(NamedTuple):
    indifference: float
    preference: float
    veto: float

class SettingsValuedType(NamedTuple):
    thresholds: ThresholdType
    alternatives: int
    criteria: int
    is_cost_threshold: float
    mode: RankingMode
    all_results: bool

class SettingsBinaryType(SettingsValuedType):
    binary_threshold: float