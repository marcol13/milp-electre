import string
import json
import time
import numpy as np
import threading
import queue

from experiments.metrics import Metrics
from experiments.benchmarks.electre_III import ElectreIII
from experiments.benchmarks.problem import ValuedProblem
from experiments.core.test_data import generate_test_data
from concurrent.futures import ThreadPoolExecutor, as_completed

from mcda.outranking.promethee import Promethee1, Promethee2, VShapeFunction
from mcda.core.matrices import AdjacencyValueMatrix, PerformanceTable
from mcda.core.relations import PreferenceStructure, Ranking


from mcdalp.core.score import Score
from mcdalp.outranking.crisp import CrispOutranking
from mcdalp.core.credibility import CredibilityMatrix
from mcdalp.outranking.ranking import Ranking as LpRanking

class NetFlowScorePartial(Promethee1):
    def __init__(self, netflow, scales, weights, thresholds):
        self.netflow = AdjacencyValueMatrix(netflow)
        self.functions = dict(zip(range(netflow.shape[0]), [VShapeFunction(p=thresholds["preference"], q=thresholds["indifference"]) for _ in range(netflow.shape[0])]))
        self.performance_table = PerformanceTable(netflow, scales=scales)
        self.time = 0

        super().__init__(self.performance_table, weights, self.functions)

    def get_rank(self):
        start_time = time.time()
        final_rank = self.rank(self.netflow)
        self.time = time.time() - start_time
        return final_rank.outranking_matrix.data.to_numpy()
    
class NetFlowScoreComplete(Promethee2):
    def __init__(self, netflow, scales, weights, thresholds):
        self.netflow = AdjacencyValueMatrix(netflow)
        self.functions = dict(zip(range(netflow.shape[0]), [VShapeFunction(p=thresholds["preference"], q=thresholds["indifference"]) for _ in range(netflow.shape[0])]))
        self.performance_table = PerformanceTable(netflow, scales=scales)
        self.time = 0

        super().__init__(self.performance_table, weights, self.functions)

    def get_rank(self):
        start_time = time.time()
        flow = self.flows(self.netflow)
        ranking = Ranking(flow)
        ps= PreferenceStructure()
        transformed_rank = ps.from_ranking(ranking)
        self.time = time.time() - start_time
        return transformed_rank.outranking_matrix.data.to_numpy()


def binarize_netflow(netflow, threshold):
    return np.where(netflow > threshold, 1, 0)

def compare_netflow_score(runs: int, settings):
    labels = list(string.ascii_lowercase[:settings["alternatives"]])
    criteria_type = np.random.rand(settings["criteria"]) > settings["is_cost_threshold"]

    vp = ValuedProblem("test", settings["alternatives"], settings["criteria"], settings["thresholds"], criteria_type, labels)
    electre3 = ElectreIII(vp)

    credibility = electre3.get_matrix()
    credibility = binarize_netflow(credibility, settings["binary_threshold"])

    NetFlowScoreMethod = NetFlowScorePartial if settings["mode"] == "partial" else NetFlowScoreComplete
    netflow = NetFlowScoreMethod(credibility, electre3.scale, electre3.weights, vp.thresholds)
    
    score = Score()
    c_matrix = CredibilityMatrix(credibility)

    lp_netflowscore = CrispOutranking(c_matrix, score, labels)
    lp_netflowscore.solve(settings["mode"], all_results=False)

    rank_lp_netflow = lp_netflowscore.get_rankings()[0]
    rank_netflow = LpRanking("crisp", netflow.get_rank(), c_matrix, labels, score)

    metrics = Metrics(rank_lp_netflow, rank_netflow, settings["mode"])
    return metrics, netflow.time, lp_netflowscore.time

def run_experiment(settings, q):
    try:
        labels = list(string.ascii_lowercase[:settings["alternatives"]])
        criteria_type = np.random.rand(settings["criteria"]) > settings["is_cost_threshold"]

        vp = ValuedProblem("test", settings["alternatives"], settings["criteria"], settings["thresholds"], criteria_type, labels)

        metrics, time_comp, time_lp = compare_netflow_score(vp, settings)
        result = {**metrics.make_measurement(), "time_comp": time_comp, "time_lp": time_lp}
        q.put(result)
    except Exception as e:
        q.put(e)

def make_experiments(runs: int, settings):
    results = []
    timeout = 120

    def run_with_timeout():
        q = queue.Queue()
        thread = threading.Thread(target=run_experiment, args=(settings, q))
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            print(f"Experiment exceeded the time limit of {timeout} seconds and was terminated.")
            return None
        else:
            try:
                result = q.get_nowait()
                if isinstance(result, Exception):
                    print(f"An error occurred: {result}")
                    return None
                else:
                    print(f"Experiment completed successfully.")
                    return result
            except queue.Empty:
                print(f"Experiment completed but no result was returned.")
                return None

    with ThreadPoolExecutor(max_workers=runs) as executor:
        futures = [executor.submit(run_with_timeout) for _ in range(runs)]
        
        for future in as_completed(futures):
            try:
                result = future.result()  # No timeout here because the timeout is handled inside run_with_timeout
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                results.append(None)

    return results

def process_setting(n, setting):
    metrics = make_experiments(n, setting)
    return {
        "alternatives": setting["alternatives"],
        "criteria": setting["criteria"],
        "thresholds": setting["thresholds"],
        "is_cost_threshold": setting["is_cost_threshold"],
        "binary_threshold": setting["binary_threshold"],
        "mode": setting["mode"],
        "measurements": metrics
    }

if __name__ == "__main__":
    default_values = {
        "is_cost_threshold": 0.5,
        "mode": "partial",
        "all_results": False,
        "binary_threshold": 0.6
    }

    settings_list = generate_test_data(["alternatives", "criteria", "thresholds"], default_values)
    results = []


    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_setting, 50, setting) for setting in settings_list]

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    with open("experiments/data/netflow/partial_thresh_06.json", "w") as f:
        json.dump(results, f)
        