import json
import time
import string
import numpy as np

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from experiments.benchmarks.problem import ValuedProblem
from experiments.metrics import Metrics
from experiments.core.test_data import generate_test_data
from experiments.core.types import SettingsValuedType
from mcda.core.scales import QuantitativeScale, PreferenceDirection
from mcda.outranking.promethee import Promethee1, VShapeFunction
from mcda.core.matrices import PerformanceTable

from mcdalp.core.credibility import ValuedCredibilityMatrix
from mcdalp.core.score import Score
from mcdalp.outranking.valued_promethee import ValuedPrometheeOutranking
from mcdalp.outranking.ranking import Ranking


class PrometheeI():
    def __init__(self, problem: ValuedProblem):
        self.problem = problem
        self.scale = self.problem.create_dict([QuantitativeScale(0, 1, PreferenceDirection.MIN if is_cost else PreferenceDirection.MAX ) for is_cost in self.problem.is_cost])
        self.weights = self.problem.create_dict(self.problem.generate_weights(self.problem.criteria))

        self.time = 0
        self.table = PerformanceTable(self.problem.data, scales=self.scale, alternatives=self.problem.labels)
        self.functions = dict(zip(range(self.problem.criteria), [VShapeFunction(p=self.problem.thresholds["preference"], q=self.problem.thresholds["indifference"]) for _ in range(self.problem.criteria)]))

        self.method = Promethee1(self.table, self.weights, self.functions)

    def get_matrix(self):
        p_preferences = self.method.partial_preferences()
        preferences = self.method.preferences(p_preferences)
        return preferences.data
    
    def get_ranking(self):
        start_time = time.time()
        final_rank = self.method.rank()
        self.time = time.time() - start_time
        return final_rank.outranking_matrix.data.to_numpy()
    

def compare_promethee1(runs: int, settings: SettingsValuedType):
    labels = list(string.ascii_lowercase[:settings["alternatives"]])
    criteria_type = np.random.rand(settings["criteria"]) > settings["is_cost_threshold"]

    vp = ValuedProblem("test", settings["alternatives"], settings["criteria"], settings["thresholds"], criteria_type, labels)
    promethee1 = PrometheeI(vp)

    score = Score()
    c_matrix = ValuedCredibilityMatrix(promethee1.get_matrix().to_numpy())

    lp_promethee1 = ValuedPrometheeOutranking(c_matrix, score, labels)
    lp_promethee1.solve(settings["mode"], all_results=settings["all_results"])

    rank_lp_promethee1 = lp_promethee1.get_rankings()[0]
    rank_promethee1 = Ranking("valued", promethee1.get_ranking(), c_matrix, labels, score)

    metrics = Metrics(rank_lp_promethee1, rank_promethee1, settings["mode"])
    return metrics, promethee1.time, lp_promethee1.time

def make_experiments(runs: int, settings: SettingsValuedType) -> list[Metrics]:
    def run_experiment(_):
        labels = list(string.ascii_lowercase[:settings["alternatives"]])
        criteria_type = np.random.rand(settings["criteria"]) > settings["is_cost_threshold"]

        vp = ValuedProblem("test", settings["alternatives"], settings["criteria"], settings["thresholds"], criteria_type, labels)
        
        metrics, time_comp, time_lp = compare_promethee1(vp, settings)
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
        "mode": "partial",
        "all_results": False
    }

    settings_list = generate_test_data(["alternatives", "criteria", "thresholds"], default_values)

    results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_setting = {executor.submit(process_setting, 100, setting): setting for setting in settings_list}
        
        for future in as_completed(future_to_setting):
            result = future.result()
            print(result)
            results.append(result)

    with open("experiments/data/prometheeI/partial.json", "w") as f:
        json.dump(results, f)
