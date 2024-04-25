import numpy as np

from .outranking import Outranking
from ..core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from ..core.const import RankingMode
from pulp import lpSum

class ValuedElectreOutranking(Outranking):
    def __init__(self, credibility, scores):
        super().__init__(credibility, scores)
        self.valued_credibility = self.credibility

    def solve_partial(self):
        self.variables = self.create_variables(["outranking", "pp", "pn", "i", "r"])

        reversed_matrix = self.valued_credibility.T

        positive_preference_matrix = np.minimum(self.valued_credibility, 1 - reversed_matrix)
        negative_preference_matrix = np.minimum(reversed_matrix, 1 - self.valued_credibility)
        indifference_matrix = np.minimum(self.valued_credibility, reversed_matrix)
        incomparible_matrix = np.minimum(1 - self.valued_credibility, 1 - reversed_matrix)

        # print("POSITIVE PREFERENCE MATRIX")
        # print(positive_preference_matrix)
        # print("NEGATIVE PREFERENCE MATRIX")
        # print(negative_preference_matrix)
        # print("INDIFFERENCE MATRIX")
        # print(indifference_matrix)
        # print("INCOMPARIBLE MATRIX")
        # print(incomparible_matrix)

        problem_relations = [{"var": self.variables["pp"], "rel": PositivePreference}, {"var": self.variables["pn"], "rel": NegativePreference}, {"var": self.variables["i"], "rel": Indifference}, {"var": self.variables["r"], "rel": Incomparible}]
        valued_relations = [{"var": positive_preference_matrix, "rel": PositivePreference}, {"var": negative_preference_matrix, "rel": NegativePreference}, {"var": indifference_matrix, "rel": Indifference}, {"var": incomparible_matrix, "rel": Incomparible}]

        self.problem += lpSum([s_relation["var"][i][j] * p_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], p_relation["rel"]) for s_relation in valued_relations for p_relation in problem_relations for [i, j] in self.upper_matrix_ids])
        self.problem = self.add_contraints(RankingMode.PARTIAL, self.problem, self.variables, self.size, self.unique_permutations)

        self.problem.solve()

    def solve_complete(self):
        self.variables = self.create_variables(["p", "z"])
        reversed_matrix = self.valued_credibility.T

        positive_preference_matrix = np.minimum(self.valued_credibility, 1 - reversed_matrix)
        negative_preference_matrix = np.minimum(reversed_matrix, 1 - self.valued_credibility)
        indifference_matrix = np.minimum(self.valued_credibility, reversed_matrix)

        valued_relations = [{"var": positive_preference_matrix, "rel": PositivePreference}, {"var": negative_preference_matrix, "rel": NegativePreference}, {"var": indifference_matrix, "rel": Indifference}]

        self.problem += lpSum([s_relation["var"][i][j] * (self.variables["p"][i][j] - self.variables["z"][i][j]) * self.scores.get_distance(s_relation["rel"], PositivePreference) + (self.variables["p"][j][i] - self.variables["z"][i][j]) * s_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], NegativePreference) + self.variables["z"][i][j] * s_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], Indifference) for [i, j] in self.upper_matrix_ids for s_relation in valued_relations])
        self.problem = self.add_contraints(RankingMode.COMPLETE, self.problem, self.variables, self.size, self.unique_permutations)

        self.problem.solve()
