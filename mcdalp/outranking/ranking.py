import numpy as np

from typing import Literal
from ..core.score import Score
from ..core.types import InputType
from ..core.visualize.table.table import Table
from ..core.visualize.table.crisp_table import CrispTable
from ..core.visualize.table.valued_table import ValuedTable
from ..core.visualize.table.stochastic_table import StochasticTable
from ..core.visualize.graph.graph import Graph

class Ranking:
    def __init__(self, input_type: InputType, outranking: np.array, credibility: np.array, labels: list[str], scores: Score):
        self.input_type = input_type
        self.outranking = outranking
        self.credibility = credibility
        self.labels = labels
        self.scores = scores

        # add support and weaknesses, etc.
        self.weakness = np.sum(self.outranking, axis=0) - np.diag(self.outranking)
        self.strength = np.sum(self.outranking, axis=1) - np.diag(self.outranking)
        self.quality = self.strength - self.weakness
        self.outranked_variants = self.__get_outranked_variants_number()
        self.leaders = np.where(self.outranked_variants == 0)


    def create_table(self) -> list[Table]:
        table_dict = {
            "crisp": CrispTable,
            "valued": ValuedTable,
            "stochastic": StochasticTable
        }
        TableType = table_dict[self.input_type]

        return TableType(self.credibility, self.outranking, self.labels)
            

    def create_graph(self) -> Graph:
        return Graph(self.outranking, self.labels)

    def __get_outranked_variants_number(self):
        indifference_num = np.sum(np.logical_and(self.outranking.T, self.outranking), axis=0) - np.diag(self.outranking)
        outranked_variants = self.weakness - indifference_num
        return outranked_variants

