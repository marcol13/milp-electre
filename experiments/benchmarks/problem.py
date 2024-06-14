from typing import NamedTuple, Tuple
import numpy as np

from mcdalp.core.types import RankingMode



class ThresholdType(NamedTuple):
    indifference: float
    preference: float
    veto: float

class SettingsType(NamedTuple):
    thresholds: ThresholdType
    alternatives: int
    criteria: int
    is_cost_threshold: float
    mode: RankingMode
    all_results: bool

class Problem():
    def __init__(self, name: str, alternatives: int, criteria: int, thresholds: ThresholdType, is_cost: list[bool], labels=list[str]):
        self.name = name
        self.alternatives = alternatives
        self.criteria = criteria
        self.thresholds = thresholds
        self.is_cost = is_cost
        self.labels = labels

        assert len(is_cost) == criteria
        assert len(labels) == alternatives

    def _generate_outranking(self):
        matrix = np.random.rand(self.alternatives, self.criteria)
        return matrix

class BinaryProblem(Problem):
    def __init__(self, name: str, alternatives: int, criteria: int, thresholds: ThresholdType, is_cost: list[bool], labels: list[str], binary_threshold: float):
        super().__init__(name, alternatives, criteria, thresholds, is_cost, labels)
        self.binary_threshold = binary_threshold
        self.data = self.__binarize_data()

    def __binarize_data(self):
        data = self._generate_outranking()
        data = np.where(data > self.binary_threshold, 1, 0)
        return data

class ValuedProblem(Problem):
    def __init__(self, name: str, alternatives: int, criteria: int, thresholds: ThresholdType, is_cost: list[bool], labels: list[str]):
        super().__init__(name, alternatives, criteria, thresholds, is_cost, labels)
        self.data = self._generate_outranking()
