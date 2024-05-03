from .outranking import Outranking
from ..core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from ..core.const import RankingMode
from ..core.visualize.table import StochasticTable
from pulp import lpSum

class StochasticOutranking(Outranking):
    def __init__(self, credibility, scores, labels):
        super().__init__(credibility, scores, labels)
        self.stochastic_credibility = self.credibility

    def init_partial(self, problem):
        variables = self.create_variables(["outranking", "pp", "pn", "i", "r"])
        problem_relations = [{"var": variables["pp"], "rel": PositivePreference}, {"var": variables["pn"], "rel": NegativePreference}, {"var": variables["i"], "rel": Indifference}, {"var": variables["r"], "rel": Incomparible}]
        stochastic_relations = [{"var": self.stochastic_credibility[PositivePreference], "rel": PositivePreference}, {"var": self.stochastic_credibility[NegativePreference], "rel": NegativePreference}, {"var": self.stochastic_credibility[Indifference], "rel": Indifference}, {"var": self.stochastic_credibility[Incomparible], "rel": Incomparible}]

        problem += lpSum([s_relation["var"][i][j] * p_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], p_relation["rel"]) for s_relation in stochastic_relations for p_relation in problem_relations for [i, j] in self.upper_matrix_ids])
        problem = self.add_contraints(RankingMode.PARTIAL, problem, variables, self.size, self.unique_permutations)

        return problem

    def init_complete(self, problem):
        variables = self.create_variables(["p", "z"])
        stochastic_relations = [{"var": self.stochastic_credibility[PositivePreference], "rel": PositivePreference}, {"var": self.stochastic_credibility[NegativePreference], "rel": NegativePreference}, {"var": self.stochastic_credibility[Indifference], "rel": Indifference}]

        problem += lpSum([s_relation["var"][i][j] * (variables["p"][i][j] - variables["z"][i][j]) * self.scores.get_distance(s_relation["rel"], PositivePreference) + (variables["p"][j][i] - variables["z"][i][j]) * s_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], NegativePreference) + variables["z"][i][j] * s_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], Indifference) for [i, j] in self.upper_matrix_ids for s_relation in stochastic_relations])
        problem = self.add_contraints(RankingMode.COMPLETE, problem, variables, self.size, self.unique_permutations)

        return problem
    
    def create_table(self, all_results: bool = False):
        if all_results:
            for result in self.results:
                table = StochasticTable(self.credibility, result, self.labels)
                table.draw()
        else:
            table = StochasticTable(self.credibility, self.results[0], self.labels)
            table.draw()
