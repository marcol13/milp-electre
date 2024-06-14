import string
import numpy as np

from tqdm import tqdm
from experiments.benchmarks.problem import ValuedProblem, SettingsType
from experiments.metrics import kendall_tau, kendall_distance, normalized_hit_ratio
from mcda.core.scales import QuantitativeScale, PreferenceDirection
from mcda.outranking.promethee import Promethee1, VShapeFunction
from mcda.core.matrices import PerformanceTable

from mcdalp.core.credibility import ValuedCredibilityMatrix
from mcdalp.core.score import Score
from mcdalp.outranking.valued_promethee import ValuedPrometheeOutranking
from mcdalp.outranking.ranking import Ranking


class PrometheeII():
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
    

def compare_promethee2(runs: int, settings: SettingsType):
    results = []
    print(settings)
    for _ in tqdm(range(runs)):
        labels = list(string.ascii_lowercase[:settings["alternatives"]])
        criteria_type = np.random.rand(settings["criteria"]) > settings["is_cost_threshold"]

        vp = ValuedProblem("test", settings["alternatives"], settings["criteria"], settings["thresholds"], criteria_type, labels)
        promethee1 = PrometheeII(vp)

        score = Score()
        c_matrix = ValuedCredibilityMatrix(promethee1.get_matrix().to_numpy())
        

        lp_promethee1 = ValuedPrometheeOutranking(c_matrix, score, labels)
        lp_promethee1.solve(settings["mode"], all_results=settings["all_results"])

        rank_lp_promethee1 = lp_promethee1.get_rankings()
        rank_promethee1 = Ranking("valued", promethee1.method.rank().outranking_matrix.data.to_numpy(), c_matrix, labels, score)

        temp_results = []
        for rank in rank_lp_promethee1:
            distance = kendall_distance(rank.outranking, rank_promethee1.outranking)
            kendall = kendall_tau(distance, rank.outranking.shape[0])
            nhr = normalized_hit_ratio(rank_promethee1, rank)
            temp_results.append((kendall, nhr))

        temp_results = np.array(temp_results)
        results.append(np.average(temp_results, axis=0))

    return np.array(results)

if __name__ == "__main__":
    settings = {
        "alternatives": 8,
        "criteria": 5,
        "thresholds": {
            "indifference": 0.05,
            "preference": 0.15,
            "veto": 0.25
        },
        "is_cost_threshold": 0.5,
        "mode": "complete",
        "all_results": True
    }

    metrics = compare_promethee2(10, settings)
    print(metrics)