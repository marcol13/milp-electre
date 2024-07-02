import string
import numpy as np

from tqdm import tqdm
from experiments.benchmarks.problem import ValuedProblem
from experiments.core.types import SettingsValuedType
from experiments.core.test_data import generate_test_data
from experiments.metrics import Metrics
from mcda.core.scales import QuantitativeScale, PreferenceDirection
from mcda.core.relations import PreferenceStructure
from mcda.outranking.promethee import Promethee2, VShapeFunction
from mcda.core.matrices import PerformanceTable

from mcdalp.core.credibility import ValuedCredibilityMatrix
from mcdalp.core.score import Score
from mcdalp.outranking.valued_promethee import ValuedPrometheeOutranking
from mcdalp.outranking.ranking import Ranking


class PrometheeII():
    def __init__(self, problem: ValuedProblem):
        self.problem = problem
        self.scale = self.problem.create_dict([QuantitativeScale(0, 1, PreferenceDirection.MIN if is_cost else PreferenceDirection.MAX ) for is_cost in self.problem.is_cost])
        self.weights = self.problem.create_dict(self.problem.generate_weights(self.problem.criteria))
        # self.scale = dict(zip(range(self.problem.criteria), [QuantitativeScale(0, 1, PreferenceDirection.MIN if is_cost else PreferenceDirection.MAX ) for is_cost in self.problem.is_cost]))
        # self.weights = dict(zip(range(self.problem.criteria), [1 for _ in range(self.problem.criteria)]))
        
        self.table = PerformanceTable(self.problem.data, scales=self.scale, alternatives=self.problem.labels)
        self.functions = dict(zip(range(self.problem.criteria), [VShapeFunction(p=self.problem.thresholds["preference"], q=self.problem.thresholds["indifference"]) for _ in range(self.problem.criteria)]))

        self.method = Promethee2(self.table, self.weights, self.functions)

    def get_matrix(self):
        p_preferences = self.method.partial_preferences()
        preferences = self.method.preferences(p_preferences)
        return preferences.data
    
    def get_credibility(self):
        rank = self.method.rank()
        ps = PreferenceStructure()
        transformed_rank = ps.from_ranking(rank)

        return transformed_rank.outranking_matrix.data.to_numpy()
    

def compare_promethee2(runs: int, settings: SettingsValuedType):
    results = []
    for _ in tqdm(range(runs)):
        labels = list(string.ascii_lowercase[:settings["alternatives"]])
        criteria_type = np.random.rand(settings["criteria"]) > settings["is_cost_threshold"]

        vp = ValuedProblem("test", settings["alternatives"], settings["criteria"], settings["thresholds"], criteria_type, labels)
        promethee2 = PrometheeII(vp)

        score = Score()
        c_matrix = ValuedCredibilityMatrix(promethee2.get_matrix().to_numpy())
        
        lp_promethee2 = ValuedPrometheeOutranking(c_matrix, score, labels)
        lp_promethee2.solve(settings["mode"], all_results=settings["all_results"])
        
        # In experiments there is checked only the first ranking
        rank_lp_promethee2 = lp_promethee2.get_rankings()[0]
        rank_promethee2 = Ranking("valued", promethee2.get_credibility(), c_matrix, labels, score)

        metrics = Metrics(rank_lp_promethee2, rank_promethee2, settings["mode"])
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
    #     "all_results": True
    # }

    default_values = {
        "is_cost_threshold": 0.5,
        "mode": "partial",
        "all_results": False
    }

    settings_list = generate_test_data(["alternatives", "criteria", "thresholds"], default_values)

    for setting in settings_list:
        metrics = compare_promethee2(10, setting)
        print(metrics)
