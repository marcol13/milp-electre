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
        self.outranking = np.asarray(outranking)
        self.outranking = np.reshape(self.outranking, (-1, *self.outranking.shape[1:]))
        self.credibility = credibility
        self.labels = labels
        self.scores = scores

        # TODO: sort it out
        self.positions = np.sum(self.outranking, axis=1)

    def create_tables(self) -> list[Table]:
        matrices = []
        table_dict = {
            "crisp": CrispTable,
            "valued": ValuedTable,
            "stochastic": StochasticTable
        }
        TableType = table_dict[self.input_type]
        for rank in self.outranking:
            matrices.append(TableType(self.credibility, rank, self.labels))

        return matrices
            

    def create_graphs(self) -> list[Graph]:
        graphs = []
        for rank in self.outranking:
            graphs.append(Graph(rank, self.labels))

        return graphs


