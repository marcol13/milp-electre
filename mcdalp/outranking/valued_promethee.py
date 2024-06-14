from .outranking import Outranking
from ..core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from ..core.const import RankingMode
from ..core.visualize.table.valued_table import ValuedTable
from pulp import lpSum

class ValuedPrometheeOutranking(Outranking):
    def __init__(self, credibility, scores, labels):
        super().__init__(credibility, scores, labels)
        self.valued_credibility = self.credibility

    def init_partial(self, problem):
        variables = self.create_variables(["outranking", "pp", "pn", "i", "r"])
        reversed_matrix = self.valued_credibility.T
        problem_relations = [{"var": variables["pp"], "rel": PositivePreference}, {"var": variables["pn"], "rel": NegativePreference}, {"var": variables["i"], "rel": Indifference}, {"var": variables["r"], "rel": Incomparible}]
        valued_relations = [{"var": self.valued_credibility, "rel": PositivePreference}, {"var": reversed_matrix, "rel": NegativePreference}, {"var": 1 - self.valued_credibility - reversed_matrix, "rel": Indifference}]

        problem += lpSum([s_relation["var"][i][j] * p_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], p_relation["rel"]) for s_relation in valued_relations for p_relation in problem_relations for [i, j] in self.upper_matrix_ids])
        problem = self.add_contraints(RankingMode.PARTIAL, problem, variables, self.size, self.unique_permutations)

        return problem

    def init_complete(self, problem):
        variables = self.create_variables(["p", "z"])
        reversed_matrix = self.valued_credibility.T
        valued_relations = [{"var": self.valued_credibility, "rel": PositivePreference}, {"var": reversed_matrix, "rel": NegativePreference}, {"var": 1 - self.valued_credibility - reversed_matrix, "rel": Indifference}]

        problem += lpSum([s_relation["var"][i][j] * (variables["p"][i][j] - variables["z"][i][j]) * self.scores.get_distance(s_relation["rel"], PositivePreference) + (variables["p"][j][i] - variables["z"][i][j]) * s_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], NegativePreference) + variables["z"][i][j] * s_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], Indifference) for [i, j] in self.upper_matrix_ids for s_relation in valued_relations])
        problem = self.add_contraints(RankingMode.COMPLETE, problem, variables, self.size, self.unique_permutations)

        return problem
    
    def create_table(self, all_results: bool = False):
        if all_results:
            for result in self.results:
                table = ValuedTable(self.credibility, result, self.labels)
                table.draw()
        else:
            table = ValuedTable(self.credibility, self.results[0], self.labels)
            table.draw()
