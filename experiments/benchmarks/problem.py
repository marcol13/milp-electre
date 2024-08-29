import numpy as np

from experiments.core.types import ThresholdType

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
    
    def generate_weights(self, n: int):
        bins = np.random.random_sample(n-1)
        bins = np.append(bins, 0)
        bins = np.append(bins, 1)
        bins.sort()
        weights = np.diff(bins)
        return weights
    
    def create_dict(self, data):
        assert len(data) == self.criteria
        return dict(zip(range(self.criteria), data))


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
