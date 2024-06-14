import numpy as np
from metrics import kendall_tau, kendall_distance
from mcdalp.outranking.ranking import Ranking

matrix_a = np.array([[0,1,1],[0,0,1],[0,0,0]])
matrix_b = np.array([[0,0,0],[1,0,0],[1,1,0]])

dist = kendall_distance(matrix_a, matrix_a)
score = kendall_tau(dist, matrix_a.shape[0])
# print(kendall_tau(matrix_a, matrix_b))


from mcdalp.core.credibility import CredibilityMatrix
from mcdalp.core.score import Score
from mcdalp.core.visualize.graph.graph import Graph
from mcdalp.outranking.crisp import CrispOutranking

simple1 = np.array([[1,1,1,0],[0,1,0,0],[1,1,1,0],[0,0,0,0]], dtype=np.int32)
# simple1 = np.array([[1,1,0,0],[0,1,1,0],[1,0,1,0],[0,0,0,0]], dtype=np.int32)
# simple1 = np.array([[1,1,0],[0,1,1],[1,0,1]], dtype=np.int32)
l_simple1 = ["A", "B", "C", "D"]
c_simple1 = CredibilityMatrix(simple1)
s_simple1 = Score()

r_simple1 = CrispOutranking(c_simple1, s_simple1, l_simple1)
r_simple1.solve("partial", all_results=True)
rankings = r_simple1.get_rankings()
print(rankings)
# for idx, rank in enumerate(rankings):
#     graph = rank.create_graph()
#     print(graph)
#     graph.show(filename=f"{idx}_digraph")

# for result in r_simple1.results:
#     print(result)
# rank = Ranking("crisp", r_simple1.results, simple1, l_simple1, s_simple1)

# graphs = rank.create_graphs()
# graphs[0].show()

# tables = rank.create_tables()
# tables[0].draw()