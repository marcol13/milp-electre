import string
import numpy as np

from experiments.metrics import kendall_tau, kendall_distance, normalized_hit_ratio, rdm, second_kendall
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
        # self.functions = {0: VShapeFunction(p=thresholds["preference"], q=thresholds["indifference"])}
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
        netflow = np.random.rand(settings["alternatives"], settings["alternatives"])
        netflow = binarize_netflow(netflow, settings["binary_threshold"])
        scales = {0: QuantitativeScale(0, 1, PreferenceDirection.MIN)}
        weights = {0: 1}
        netflowscore_method = None
        if settings["mode"] == "partial":
            netflowscore_method = NetFlowScorePartial(netflow, scales, weights, settings["thresholds"])
            # pass
            
        elif settings["mode"] == "complete":
            netflowscore_method = NetFlowScoreComplete(netflow, scales, weights, settings["thresholds"])

        score = Score()
        c_matrix = CredibilityMatrix(netflow)

        lp_netflowscore = CrispOutranking(c_matrix, score, labels)
        lp_netflowscore.solve(settings["mode"], all_results=settings["all_results"])

        rank_lp_netflow = lp_netflowscore.get_rankings()
        
        # temp = np.random.rand(settings["alternatives"], settings["alternatives"])
        # temp = binarize_netflow(temp, settings["binary_threshold"])
        rank_netflow = LpRanking("crisp", netflowscore_method.get_rank(), c_matrix, labels, score)
        # rank_netflow = LpRanking("crisp", temp, c_matrix, labels, score)

        temp_results = []
        for rank in rank_lp_netflow:
            distance = kendall_distance(rank.outranking, rank_netflow.outranking)
            kendall = kendall_tau(distance, rank.outranking.shape[0])
            nhr = normalized_hit_ratio(rank_netflow, rank)
            rank_difference = rdm(rank, rank_netflow, "partial")
            temp_results.append((kendall, nhr, rank_difference))

        temp_results = np.array(temp_results)
        results.append(np.average(temp_results, axis=0))

    return np.array(results)

if __name__ == "__main__":
    # from mcda.core.scales import QuantitativeScale, PreferenceDirection

    # is_cost = [True, True, False, False]
    # netflow = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1], [1, 0, 1, 0, 1]])
    # scales = {0: QuantitativeScale(0, 1, PreferenceDirection.MIN)}
    # weights = {0: 1}
    # thresholds = {"preference": 0.5, "indifference": 0.2}

    # score = NetFlowScorePartial(netflow, scales, weights, thresholds)
    # score = NetFlowScoreComplete(netflow, scales, weights, thresholds)
    # print(score.get_rank())
    settings = {
        "alternatives": 8,
        "criteria": 5,
        "thresholds": {
            "indifference": 0.05,
            "preference": 0.15,
            "veto": 0.25
        },
        "is_cost_threshold": 0.5,
        "mode": "partial",
        "all_results": True,
        "binary_threshold": 0.5
    }

    metrics = compare_netflow_score(10, settings)
    print(metrics)