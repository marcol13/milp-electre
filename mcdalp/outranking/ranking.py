import numpy as np
import numpy.typing as npt
import string

from typing import Union
from ..core.score import Score
from ..core.types import InputType
from ..core.visualize.table.table import Table
from ..core.visualize.table.crisp_table import CrispTable
from ..core.visualize.table.valued_table import ValuedTable
from ..core.visualize.table.stochastic_table import StochasticTable
from ..core.visualize.graph.graph import Graph

class Ranking:
    def __init__(self, input_type: InputType, rank_matrix: npt.ArrayLike, outranking: Union[npt.ArrayLike, None] = None, labels: Union[list[str], None] = None, scores: Score = Score()):
        self.input_type = input_type
        self.rank_matrix = rank_matrix
        self.outranking = outranking if outranking != None else rank_matrix
        self.size = rank_matrix.shape[0]
        self.labels = labels if labels != None else list(string.ascii_letters[:self.size])
        self.scores = scores

        # add support and weaknesses, etc.
        self.weakness = np.sum(self.rank_matrix, axis=0) - np.diag(self.rank_matrix)
        self.strength = np.sum(self.rank_matrix, axis=1) - np.diag(self.rank_matrix)
        self.quality = self.strength - self.weakness
        self.positions = self.__get_positions()
        self.leaders = np.where(self.positions == 0)


    def create_table(self) -> list[Table]:
        table_dict = {
            "crisp": CrispTable,
            "valued": ValuedTable,
            "stochastic": StochasticTable
        }
        TableType = table_dict[self.input_type]

        return TableType(self.outranking, self.rank_matrix, self.labels)
            

    def create_graph(self) -> Graph:
        return Graph(self.rank_matrix, self.labels)

    def __get_positions(self):
        indifference_num = np.sum(np.logical_and(self.rank_matrix.T, self.rank_matrix), axis=0) - np.diag(self.rank_matrix)
        outranked_variants = self.weakness - indifference_num
        return outranked_variants

