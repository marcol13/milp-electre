from .outranking import Outranking
from ..core.const import RELATIONS
from ..core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from ..core.const import RankingMode
from pulp import lpSum

class CrispOutranking(Outranking):
    def __init__(self, credibility, scores):
        super().__init__(credibility, scores)

    def solve_partial(self):
        self.variables = self.create_variables(["outranking", "pp", "pn", "i", "r"])
        problem_relations = [{"var": self.variables["pp"], "rel": PositivePreference}, {"var": self.variables["pn"], "rel": NegativePreference}, {"var": self.variables["i"], "rel": Indifference}, {"var": self.variables["r"], "rel": Incomparible}]

        self.problem += lpSum([relation["var"][i][j] * self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), relation["rel"]) for [i, j] in self.upper_matrix_ids for relation in problem_relations])
        self.problem = self.add_contraints(RankingMode.PARTIAL, self.problem, self.variables, self.size, self.unique_permutations)

        self.problem.solve()

    def solve_complete(self):
        self.variables = self.create_variables(["p", "z"])

        self.problem += lpSum([self.variables["p"][i][j] * self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), PositivePreference) + self.variables["p"][j][i] * self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), NegativePreference) + self.variables["z"][i][j] * (self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), Indifference) - self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), PositivePreference) - self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), NegativePreference)) for [i, j] in self.upper_matrix_ids])  
        self.problem = self.add_contraints(RankingMode.COMPLETE, self.problem, self.variables, self.size, self.unique_permutations)

        self.problem.solve()
        