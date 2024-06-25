from experiments.metrics import *
from mcdalp.core.score import Score
from mcdalp.outranking.ranking import Ranking

arr1 = np.array([[0,1,1],[0,0,1],[0,0,0]])
arr2 = np.array([[0,0,0],[1,0,0],[1,1,0]])
arr3 = np.array([[0,1,1],[0,0,1],[0,0,0]])
arr4 = np.array([[0,1,0],[0,0,0],[0,0,0]])

arr1_1 = np.array([[0,1,1,1], [1,0,1,1], [1,1,0,1], [0,0,0,0]])
arr1_2 = np.array([[0,0,0,0], [1,0,0,0], [0,0,0,0], [0,0,1,0]])

rank1 = Ranking("crisp", arr1_1, arr1_1, ["A", "B", "C"], Score())
rank2 = Ranking("crisp", arr1_2, arr1_2, ["A", "B", "C"], Score())

dist = kendall_distance(rank1.outranking, rank2.outranking)
score = kendall_tau(dist, arr1_1.shape[0])
nhr = normalized_hit_ratio(rank1, rank2)
rdm_score = rdm(rank1, rank2, "partial")
sec = second_kendall(rank1, rank2)

print(score)
print(nhr)
print(rdm_score)
print(sec)