import string
import numpy as np

from experiments.metrics import Metrics
from experiments.benchmarks.electre_III import ElectreIII
from experiments.benchmarks.problem import ValuedProblem
from experiments.core.test_data import generate_test_data
from tqdm import tqdm

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

        super().__init__(self.performance_table, weights, self.functions)

    def get_rank(self):
        pos_flow = self.flows(self.netflow)
        neg_flow = self.flows(self.netflow, negative=True)

        res = PreferenceStructure()
        for i, a in enumerate(self.performance_table.alternatives):
            for b in self.performance_table.alternatives[(i + 1) :]:
                res += self._flow_intersection(
                    a, b, pos_flow[a], pos_flow[b], neg_flow[a], neg_flow[b]
                )
        return res.outranking_matrix.data
    
class NetFlowScoreComplete(Promethee2):
    def __init__(self, netflow, scales, weights, thresholds):
        self.netflow = AdjacencyValueMatrix(netflow)
        self.functions = dict(zip(range(netflow.shape[0]), [VShapeFunction(p=thresholds["preference"], q=thresholds["indifference"]) for _ in range(netflow.shape[0])]))
        # self.functions = {0: VShapeFunction(p=thresholds["preference"], q=thresholds["indifference"])}
        self.performance_table = PerformanceTable(netflow, scales=scales)

        super().__init__(self.performance_table, weights, self.functions)

    def get_rank(self):
        flow = self.flows(self.netflow)
        ranking = Ranking(flow)
        ps= PreferenceStructure()
        transformed_rank = ps.from_ranking(ranking)
        return transformed_rank.outranking_matrix.data.to_numpy()


def binarize_netflow(netflow, threshold):
    return np.where(netflow > threshold, 1, 0)

def compare_netflow_score(runs: int, settings):
    results = []
    for _ in tqdm(range(runs)):
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
        lp_netflowscore.solve(settings["mode"], all_results=settings["all_results"])

        # In experiments there is checked only the first ranking
        rank_lp_netflow = lp_netflowscore.get_rankings()[0]
        rank_netflow = LpRanking("crisp", netflow.get_rank(), c_matrix, labels, score)

        metrics = Metrics(rank_lp_netflow, rank_netflow, settings["mode"])
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
    #     "mode": "partial",
    #     "all_results": True,
    #     "binary_threshold": 0.5
    # }

    default_values = {
        "is_cost_threshold": 0.5,
        "mode": "partial",
        "all_results": False
    }

    settings_list = generate_test_data(["alternatives", "criteria", "thresholds", "binary_threshold"], default_values)

    for setting in settings_list:
        metrics = compare_netflow_score(10, setting)
        print(metrics)