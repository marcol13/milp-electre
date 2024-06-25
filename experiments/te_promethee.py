import numpy as np

from benchmarks.problem import ValuedProblem
from benchmarks.promethee_I import PrometheeI
from mcdalp.outranking.valued_promethee import ValuedPrometheeOutranking
from mcdalp.core.credibility import ValuedCredibilityMatrix
from mcdalp.core.score import Score
from mcdalp.outranking.ranking import Ranking
from metrics import kendall_tau, kendall_distance, normalized_hit_ratio


thresholds = {
    "indifference": 0.05,
    "preference": 0.15,
    "veto": 0.25
}
is_cost = np.random.rand(5) > 0.5
vp = ValuedProblem("test", 8, 5, thresholds, is_cost, ["a", "b", "c", "d", "e", "f", "g", "h"])

pi = PrometheeI(vp)
print("XDDDDD")
c_matrix = ValuedCredibilityMatrix(pi.get_matrix().to_numpy())


score = Score()
vp_new = ValuedPrometheeOutranking(c_matrix, score, ["a", "b", "c", "d", "e", "f", "g", "h"])
vp_new.solve("partial", all_results=True)
ranking = vp_new.get_rankings()
print(ranking)

pi_outranking= pi.method.rank().outranking_matrix.data.to_numpy()
pi_ranking = Ranking("valued", pi_outranking, c_matrix, ["a", "b", "c", "d", "e", "f", "g", "h"], score)
# print(pi.method.rank().outranking_matrix.data)

print("LOL")
print(ranking[0].outranking)
print(pi_ranking.outranking.shape, ranking[0].outranking.shape)
distance = kendall_distance(pi_ranking.outranking, ranking[0].outranking)
score = kendall_tau(distance, pi_ranking.outranking.shape[0])
print(score)

nhr = normalized_hit_ratio(pi_ranking, ranking[0])
print(nhr)


# print(vp_new.get_rankings())

# print(pi.method.partial_preferences())