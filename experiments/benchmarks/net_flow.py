import string
import json
import time
import numpy as np
import threading
import queue
import multiprocessing

from experiments.metrics import Metrics
from experiments.benchmarks.electre_III import ElectreIII
from experiments.benchmarks.problem import ValuedProblem
from experiments.core.test_data import generate_test_data
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from mcda.outranking.promethee import Promethee1, Promethee2, VShapeFunction
from mcda.core.matrices import AdjacencyValueMatrix, PerformanceTable
from mcda.core.relations import PreferenceStructure, Ranking
from mcda.core.scales import QuantitativeScale, PreferenceDirection


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
    # def get_rank(self):
    #     pos_flow = self.flows(self.netflow)
    #     neg_flow = self.flows(self.netflow, negative=True)

    #     res = PreferenceStructure()
    #     for i, a in enumerate(self.performance_table.alternatives):
    #         for b in self.performance_table.alternatives[(i + 1) :]:
    #             res += self._flow_intersection(
    #                 a, b, pos_flow[a], pos_flow[b], neg_flow[a], neg_flow[b]
    #             )
    #     return res.outranking_matrix.data
    
class NetFlowScoreComplete(Promethee2):
    def __init__(self, netflow, scales, weights, thresholds):
        self.netflow = AdjacencyValueMatrix(netflow)
        self.functions = dict(zip(range(netflow.shape[0]), [VShapeFunction(p=thresholds["preference"], q=thresholds["indifference"]) for _ in range(netflow.shape[0])]))
        # self.functions = {0: VShapeFunction(p=thresholds["preference"], q=thresholds["indifference"])}
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
    # results = []
    # for _ in tqdm(range(runs)):
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

    # In experiments there is checked only the first ranking
    rank_lp_netflow = lp_netflowscore.get_rankings()[0]
    rank_netflow = LpRanking("crisp", netflow.get_rank(), c_matrix, labels, score)

    metrics = Metrics(rank_lp_netflow, rank_netflow, settings["mode"])
    # results.append(metrics.make_measurement())
    return metrics, netflow.time, lp_netflowscore.time

    # return results


# def run_experiment(settings):
#     labels = list(string.ascii_lowercase[:settings["alternatives"]])
#     criteria_type = np.random.rand(settings["criteria"]) > settings["is_cost_threshold"]

#     vp = ValuedProblem("test", settings["alternatives"], settings["criteria"], settings["thresholds"], criteria_type, labels)

#     metrics, time_comp, time_lp = compare_netflow_score(vp, settings)
#     return {**metrics.make_measurement(), "time_comp": time_comp, "time_lp": time_lp}

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

# def make_experiments(runs: int, settings):
#     results = []
#     timeout = 120  # Czas limitu ustawiony na 2 minuty (120 sekund)


#     # def run_with_timeout(index):
#     #     try:
#     #         with Pool(processes=1) as pool:
#     #             result = pool.apply_async(run_experiment, (index, settings))
#     #             return result.get(timeout=timeout)
#     #     except TimeoutError:
#     #         print(f"Zadanie {index} przekroczyło limit czasu {timeout} sekund i zostało przerwane.")
#     #         return None
        
#     with ProcessPoolExecutor() as executor:
#         futures = [executor.submit(run_experiment, settings) for _ in range(runs)]
        
#         for future in as_completed(futures):
#             try:
#                 result = future.result(timeout=timeout)
#                 results.append(result)
#             except TimeoutError:
#                 print(f"An experiment exceeded the time limit of {timeout} seconds and was terminated.")
#                 # Optional: handle the timeout case, e.g., log the error

#     return results

def make_experiments(runs: int, settings):
    results = []
    timeout = 120  # Timeout set to 2 minutes (120 seconds)

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

    # def run_experiment(_):
    #     labels = list(string.ascii_lowercase[:settings["alternatives"]])
    #     criteria_type = np.random.rand(settings["criteria"]) > settings["is_cost_threshold"]

    #     vp = ValuedProblem("test", settings["alternatives"], settings["criteria"], settings["thresholds"], criteria_type, labels)
        
    #     metrics, time_comp, time_lp = compare_netflow_score(vp, settings)
    #     return {**metrics.make_measurement(), "time_comp": time_comp, "time_lp": time_lp}

    # results = []
    # timeout = 120  # Czas limitu ustawiony na 2 minuty (120 sekund)

    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(run_experiment, _) for _ in range(runs)]
        
    #     for future in as_completed(futures):
    #         try:
    #             result = future.result(timeout=timeout)
    #             results.append(result)
    #         except TimeoutError:
    #             print(f"Zadanie przekroczyło limit czasu {timeout} sekund i zostało przerwane.")
    #             # Możesz dodać dodatkowe działania, np. rejestrowanie nieudanych prób.

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

# def run_in_pool(setting, n, timeout):
#     try:
#         # Tworzymy nowy proces, który uruchamia funkcję process_setting
#         with Pool(processes=8) as pool:
#             result = pool.apply_async(process_setting, (setting, n))
#             return result.get(timeout=timeout)
#     except TimeoutError:
#         print(f"Zadanie dla ustawienia {setting} przekroczyło limit czasu {timeout} sekund.")
#         return None

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
    #     "mode": "partial",
    #     "all_results": True,
    #     "binary_threshold": 0.5
    # }

    default_values = {
        "is_cost_threshold": 0.5,
        "mode": "partial",
        "all_results": False,
        "binary_threshold": 0.6
    }

    settings_list = generate_test_data(["alternatives", "criteria", "thresholds"], default_values)
    # settings_list = settings_list[::-1]
    # settings_list = settings_list[:2]

    # print(settings_list)

    # raise
    results = []


    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_setting, 50, setting) for setting in settings_list]

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    with open("experiments/data/netflow/partial_thresh_06.json", "w") as f:
        json.dump(results, f)

    # with Pool(processes=multiprocessing.cpu_count()) as pool:
    #     futures = [pool.apply_async(process_setting, (5, setting)) for setting in settings_list]
        
    #     for future in futures:
    #         result = future.get()
    #         if result is not None:
    #             results.append(result)
    # with ProcessPoolExecutor(max_workers=8) as executor:
    #     future_to_setting = {executor.submit(process_setting, 5, setting): setting for setting in settings_list}
        
    #     for future in as_completed(future_to_setting):
    #         result = future.result()
    #         print(result)
    #         results.append(result)

    # with open("experiments/data/netflow/partial.json", "w") as f:
    #     json.dump(results, f)

    # for setting in settings_list:
    #     metrics = compare_netflow_score(10, setting)
    #     print(metrics)