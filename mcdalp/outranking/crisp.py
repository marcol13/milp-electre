from .outranking import Outranking
from ..core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from ..core.const import RankingMode
from ..core.visualize.table.crisp_table import CrispTable
from pulp import lpSum

class CrispOutranking(Outranking):
    def __init__(self, credibility, scores, labels):
        super().__init__(credibility, scores, labels)

    def init_partial(self, problem):
        variables = self.create_variables(["outranking", "pp", "pn", "i", "r"])
        problem_relations = [{"var": variables["pp"], "rel": PositivePreference}, {"var": variables["pn"], "rel": NegativePreference}, {"var": variables["i"], "rel": Indifference}, {"var": variables["r"], "rel": Incomparible}]

        problem += lpSum([relation["var"][i][j] * self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), relation["rel"]) for [i, j] in self.upper_matrix_ids for relation in problem_relations])
        problem = self.add_contraints(RankingMode.PARTIAL, problem, variables, self.size, self.unique_permutations)

        return problem

    def init_complete(self, problem):
        variables = self.create_variables(["p", "z"])

        problem += lpSum([variables["p"][i][j] * self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), PositivePreference) + variables["p"][j][i] * self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), NegativePreference) + variables["z"][i][j] * (self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), Indifference) - self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), PositivePreference) - self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), NegativePreference)) for [i, j] in self.upper_matrix_ids])  
        problem = self.add_contraints(RankingMode.COMPLETE, problem, variables, self.size, self.unique_permutations)

        return problem
        
    def create_table(self, all_results: bool = False):
        if all_results:
            for result in self.results:
                table = CrispTable(self.credibility, result, self.labels)
                table.draw()
        else:
            table = CrispTable(self.credibility, self.results[0], self.labels)
            table.draw()