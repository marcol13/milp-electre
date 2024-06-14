from .problem import ValuedProblem
from mcda.core.scales import QuantitativeScale, PreferenceDirection
from mcda.outranking.promethee import Promethee1, VShapeFunction
from mcda.core.matrices import PerformanceTable

class PrometheeI():
    def __init__(self, problem: ValuedProblem):
        self.problem = problem
        self.scale = dict(zip(range(self.problem.criteria), [QuantitativeScale(0, 1, PreferenceDirection.MIN if is_cost else PreferenceDirection.MAX ) for is_cost in self.problem.is_cost]))
        self.weights = dict(zip(range(self.problem.criteria), [1 for _ in range(self.problem.criteria)]))
        self.table = PerformanceTable(self.problem.data, scales=self.scale, alternatives=self.problem.labels)

        self.functions = dict(zip(range(self.problem.criteria), [VShapeFunction(p=self.problem.thresholds["preference"], q=self.problem.thresholds["indifference"]) for _ in range(self.problem.criteria)]))

        self.method = Promethee1(self.table, self.weights, self.functions)

    def get_matrix(self):
        p_preferences = self.method.partial_preferences()
        preferences = self.method.preferences(p_preferences)
        return preferences.data
    