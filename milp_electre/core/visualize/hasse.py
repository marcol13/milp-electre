import numpy as np
from .node import Node
from collections import defaultdict
# from .edge import Edge

class Hasse:
    def __init__(self, matrix: np.ndarray, labels: list[str]):
        self.matrix = matrix
        self.labels = labels
        self.size = len(matrix)
        self.__merge_indifference()
        self.max_level = np.max(np.sum(self.matrix, axis=1))
        print("max", self.max_level)

        self.nodes = self.__create_nodes()
        self.edges = self.__get_edges()

        self.__remove_self_loops()
        self.__remove_transitivity()

        print(self.edges)

    def __merge_indifference(self):
        diff = self.matrix - self.matrix.T
        merged = defaultdict(set)
        delete_idx = set()
        for idx, row in enumerate(diff):
            indices = np.argwhere((row == 0) & (np.arange(len(row)) != idx))
            if len(indices) > 0:
                merged[idx].update(*indices)
        
        for idx in merged.keys():
            indices = sorted([idx, *list(merged[idx])])
            delete_idx.update(indices[1:])
            new_label = '&'.join([self.labels[i] for i in indices])
            self.labels[idx] = new_label
        

        delete_idx = list(delete_idx)
        if len(delete_idx) > 0:
            self.matrix = np.delete(self.matrix, delete_idx, 0)
            self.matrix = np.delete(self.matrix, delete_idx, 1)
            self.size -= len(delete_idx)
            for del_idx in delete_idx[::-1]:
                self.labels.pop(del_idx)

    def __create_nodes(self):
        nodes = []
        for idx, label in enumerate(self.labels):
            outranking_level = int(np.sum(self.matrix[idx]))
            nodes.append(Node(label, outranking_level))
        return nodes

    def __get_edges(self):
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
        for pair in self.edges:
            if pair[0] == pair[1]:
                pair[0].superiors.remove(pair[0])
                self.edges.remove(pair)

    def __remove_transitivity(self):
        for node in self.nodes:
            for superior in node.superiors:
                self.__dfs_search(node, superior, [])

