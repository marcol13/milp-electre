import string
import time
import numpy as np
import json

from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

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
from mcdalp.outranking.ranking import Ranking as LpRanking


class ElectreIII():
    def __init__(self, problem: ValuedProblem):
        self.problem = problem
        self.scale = self.problem.create_dict([QuantitativeScale(0, 1, PreferenceDirection.MIN if is_cost else PreferenceDirection.MAX ) for is_cost in self.problem.is_cost])
        self.weights = self.problem.create_dict(self.problem.generate_weights(self.problem.criteria))
        self.P = self.problem.create_dict([self.problem.thresholds["preference"] for _ in range(self.problem.criteria)])
        self.I = self.problem.create_dict([self.problem.thresholds["indifference"] for _ in range(self.problem.criteria)])
        self.V = self.problem.create_dict([self.problem.thresholds["veto"] for _ in range(self.problem.criteria)])

        self.table = PerformanceTable(self.problem.data, scales=self.scale, alternatives=self.problem.labels)
        self.time = 0

        self.method = Electre3(self.table, self.weights, self.I, self.P, self.V)

    def get_matrix(self):
        credibility = self.method.credibility()
        return credibility.data
    
    def get_ranking(self):
        start_time = time.time()
        final_rank = self.method.rank()
        self.time = time.time() - start_time
        return final_rank.data.to_numpy()
    
def compare_electre3(vp: ValuedProblem, settings: SettingsValuedType) -> Tuple[Metrics, float, float]:
    electre3 = ElectreIII(vp)

    score = Score()
    c_matrix = ValuedCredibilityMatrix(electre3.get_matrix().to_numpy())
    
    lp_electre3 = ValuedElectreOutranking(c_matrix, score, vp.labels)
    lp_electre3.solve(settings["mode"], all_results=settings["all_results"])

    rank_lp_electre3 = lp_electre3.get_rankings()[0]

    if settings["mode"] == "partial":
        rank_electre3 = LpRanking("valued", electre3.get_ranking(), c_matrix, vp.labels, score)
    else:
        rank_electre3_partial = LpRanking("valued", electre3.get_ranking(), c_matrix, vp.labels, score)
        complete_preorder = rank_electre3_partial.create_rank_preorder()
        rank_electre3 = LpRanking("valued", complete_preorder, c_matrix, vp.labels, score)

    return Metrics(rank_lp_electre3, rank_electre3, settings["mode"]), electre3.time, lp_electre3.time

def make_experiments(runs: int, settings: SettingsValuedType) -> list[Metrics]:
    def run_experiment(_):
        labels = list(string.ascii_lowercase[:settings["alternatives"]])
        criteria_type = np.random.rand(settings["criteria"]) > settings["is_cost_threshold"]

        vp = ValuedProblem("test", settings["alternatives"], settings["criteria"], settings["thresholds"], criteria_type, labels)
        
        metrics, time_comp, time_lp = compare_electre3(vp, settings)
        return {**metrics.make_measurement(), "time_comp": time_comp, "time_lp": time_lp}

    results = []
    timeout = 120

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_experiment, _) for _ in range(runs)]
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except TimeoutError:
                print(f"Task exceed {timeout} seconds.")

    return results

def process_setting(n, setting):
    metrics = make_experiments(n, setting)
    return {
        "alternatives": setting["alternatives"],
        "criteria": setting["criteria"],
        "thresholds": setting["thresholds"],
        "is_cost_threshold": setting["is_cost_threshold"],
        "mode": setting["mode"],
        "measurements": metrics
    }

if __name__ == "__main__":
    default_values = {
        "is_cost_threshold": 0.5,
        "mode": "complete",
        "all_results": False
    }

    settings_list = generate_test_data(["alternatives", "criteria", "thresholds"], default_values)
    results = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_setting = {executor.submit(process_setting, 100, setting): setting for setting in settings_list}
        
        for future in as_completed(future_to_setting):
            result = future.result()
            results.append(result)

    with open("experiments/data/electreIII/complete.json", "w") as f:
        json.dump(results, f)

