import numpy as np

from collections import defaultdict
from .node import Node
from ..types import QuadraticArrayType

class Hasse:
    def __init__(self, matrix: QuadraticArrayType, labels: list[str]):
        self.matrix = matrix
        self.labels = labels
        self.size = len(matrix)
        self.__merge_indifference()

        self.nodes = self.__create_nodes()
        self.edges = self.__get_edges()

        self.__remove_self_loops()
        self.__remove_transitivity()

    #TODO: Make it a return function
    def __merge_indifference(self):
        diff = self.matrix - self.matrix.T
        merged = defaultdict(set)
        delete_idx = set()
        for idx, row in enumerate(diff):
            indices = np.argwhere((row == 0) & (np.arange(len(row)) != idx) & (self.matrix[idx] == 1))
            if len(indices) > 0:
                merged[idx].update(*indices)
        
        for idx in merged.keys():
            indices = sorted([idx, *list(merged[idx])])
            delete_idx.update(indices[1:])
            new_label = '&'.join([self.labels[i] for i in indices])
            self.labels[idx] = new_label
        

        delete_idx = sorted(list(delete_idx))
        if len(delete_idx) > 0:
            self.matrix = np.delete(self.matrix, delete_idx, 0)
            self.matrix = np.delete(self.matrix, delete_idx, 1)
            self.size -= len(delete_idx)
            for del_idx in delete_idx[::-1]:
                self.labels.pop(del_idx)

    def __create_nodes(self) -> list[Node]:
        nodes = []
        for idx, label in enumerate(self.labels):
            outranking_level = int(np.sum(self.matrix[idx]))
            nodes.append(Node(label, outranking_level))
        return nodes

    def __get_edges(self) -> list[tuple[Node, Node]]:
        edges = []
        for index in np.argwhere(self.matrix == 1):
            superior = self.nodes[index[0]]
            collateral = self.nodes[index[1]]

            collateral.add_superior(superior)
            edges.append((collateral, superior))
        return edges
    
    def __dfs_search(self, original, node, visited):
        visited.append(node)
        
        for superior in node.superiors:
            if (original, superior) in self.edges:
                self.edges.remove((original, superior))
            if superior not in visited:
                self.__dfs_search(original, superior, visited)
    
    def __remove_self_loops(self):
        delete_edges = []
        for pair in self.edges:
            if pair[0] == pair[1]:
                pair[0].superiors.remove(pair[0])
                delete_edges.append(pair)

        for pair in delete_edges:
            self.edges.remove(pair)

    def __remove_transitivity(self):
        for node in self.nodes:
            for superior in node.superiors:
                self.__dfs_search(node, superior, [])

