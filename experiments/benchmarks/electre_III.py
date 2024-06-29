import string
import numpy as np

from tqdm import tqdm
from experiments.core.types import SettingsValuedType
from experiments.benchmarks.problem import ValuedProblem
from experiments.metrics import Metrics
from experiments.core.test_data import generate_test_data
from mcda.core.scales import QuantitativeScale, PreferenceDirection
from mcda.outranking.electre import Electre3
from mcda.core.matrices import PerformanceTable

from mcdalp.core.credibility import ValuedCredibilityMatrix
from mcdalp.core.score import Score
from mcdalp.outranking.valued_electre import ValuedElectreOutranking
from mcdalp.outranking.ranking import Ranking


class ElectreIII():
    def __init__(self, problem: ValuedProblem):
        self.problem = problem
        self.scale = self.problem.create_dict([QuantitativeScale(0, 1, PreferenceDirection.MIN if is_cost else PreferenceDirection.MAX ) for is_cost in self.problem.is_cost])
        self.weights = self.problem.create_dict(self.problem.generate_weights(self.problem.criteria))
        self.P = self.problem.create_dict([self.problem.thresholds["preference"] for _ in range(self.problem.criteria)])
        self.I = self.problem.create_dict([self.problem.thresholds["indifference"] for _ in range(self.problem.criteria)])
        self.V = self.problem.create_dict([self.problem.thresholds["veto"] for _ in range(self.problem.criteria)])

        # self.scale = dict(zip(range(self.problem.criteria), [QuantitativeScale(0, 1, PreferenceDirection.MIN if is_cost else PreferenceDirection.MAX ) for is_cost in self.problem.is_cost]))
        # self.weights = dict(zip(range(self.problem.criteria), [1 for _ in range(self.problem.criteria)]))
        # self.P = dict(zip(range(self.problem.criteria), [self.problem.thresholds["preference"] for _ in range(self.problem.criteria)]))
        # self.I = dict(zip(range(self.problem.criteria), [self.problem.thresholds["indifference"] for _ in range(self.problem.criteria)]))
        # self.V = dict(zip(range(self.problem.criteria), [self.problem.thresholds["veto"] for _ in range(self.problem.criteria)]))

        self.table = PerformanceTable(self.problem.data, scales=self.scale, alternatives=self.problem.labels)

        self.method = Electre3(self.table, self.weights, self.I, self.P, self.V)

    def get_matrix(self):
        credibility = self.method.credibility()
        return credibility.data
    
    def get_credibility(self):
        credibility = self.method.credibility()

        return credibility.data.to_numpy()
    

def compare_electre3(runs: int, settings: SettingsValuedType):
    results = []
    for _ in tqdm(range(runs)):
        labels = list(string.ascii_lowercase[:settings["alternatives"]])
        criteria_type = np.random.rand(settings["criteria"]) > settings["is_cost_threshold"]

        vp = ValuedProblem("test", settings["alternatives"], settings["criteria"], settings["thresholds"], criteria_type, labels)
        electre3 = ElectreIII(vp)

        score = Score()
        c_matrix = ValuedCredibilityMatrix(electre3.get_matrix().to_numpy())
        
        lp_electre3 = ValuedElectreOutranking(c_matrix, score, labels)
        lp_electre3.solve(settings["mode"], all_results=settings["all_results"])

        # In experiments there is checked only the first ranking
        rank_lp_electre3 = lp_electre3.get_rankings()[0]
        rank_electre3 = Ranking("valued", electre3.get_credibility(), c_matrix, labels, score)

        metrics = Metrics(rank_lp_electre3, rank_electre3, settings["mode"])
        results.append(metrics.make_measurement())

    return results

if __name__ == "__main__":
    # settings = {
    #     "alternatives": 8,
    #     "criteria": 5,
    #     "thresholds": {
    #         "indifference": 0.05,
    #         "preference": 0.15,
    #         "veto": 0.25
    #     },
    #     "is_cost_threshold": 0.5,
    #     "mode": "complete",
    #     "all_results": False
    # }

    default_values = {
        "is_cost_threshold": 0.5,
        "mode": "partial",
        "all_results": False
    }

    settings_list = generate_test_data(["alternatives", "criteria", "thresholds"], default_values)

    for setting in settings_list:
        metrics = compare_electre3(10, setting)
        print(metrics)
