import numpy as np

from collections import defaultdict
from .node import Node
from .edge import Edge
from ...types import QuadraticArrayType

class Hasse:
    def __init__(self, matrix: QuadraticArrayType, labels: list[str]):
        self.matrix, self.labels = self.__merge_indifference(matrix, labels)
        self.size = len(self.matrix)

        self.nodes = self.__create_nodes(self.matrix, self.labels)
        self.edges = self.__create_edges(self.matrix, self.nodes)

        self.edges = self.__hasse_process(self.nodes, self.edges)

    def __merge_indifference(self, matrix: QuadraticArrayType, labels: list[str], sep: str="&") -> list[QuadraticArrayType, list[str]]:
        diff = matrix - matrix.T
        merged = defaultdict(set)
        del_indices = set()
        for idx, row in enumerate(diff):
            indices = np.argwhere((row == 0) & (np.arange(len(row)) != idx) & (matrix[idx] == 1))
            if len(indices) > 0:
                merged[idx].update(*indices)
        
        for idx in merged.keys():
            indices = sorted([idx, *list(merged[idx])])
            del_indices.update(indices[1:])
            new_label = sep.join([labels[i] for i in indices])
            labels[idx] = new_label
        

        del_indices = sorted(list(del_indices))
        if len(del_indices) > 0:
            matrix = np.delete(matrix, del_indices, 0)
            matrix = np.delete(matrix, del_indices, 1)
            for del_idx in del_indices[::-1]:
                labels.pop(del_idx)

        return matrix, labels

    def __create_nodes(self, matrix: QuadraticArrayType, labels: list[str]) -> list[Node]:
        nodes = []
        for idx, label in enumerate(labels):
            outranking_level = int(np.sum(matrix[idx]))
            nodes.append(Node(label, outranking_level))
        return nodes

    def __create_edges(self, matrix:QuadraticArrayType, nodes: list[Node]) -> list[Edge]:
        edges = []
        for index in np.argwhere(matrix == 1):
            superior = nodes[index[0]]
            collateral = nodes[index[1]]

            collateral.add_superior(superior)
            edges.append(Edge(collateral, superior))
        return edges
    
    def __dfs_search(self, original: Node, node: Node, edges: list[tuple[Node, Node]], visited: list[Node]) -> list[tuple[Node, Node]]:
        visited.append(node)
        
        for superior in node.superiors:
            if Edge(original, superior) in edges:
                edges.remove(Edge(original, superior))
            if superior not in visited:
                self.__dfs_search(original, superior, edges, visited)
        return edges
    
    def __remove_self_loops(self, edges: list[Edge]) -> list[Edge]:
        del_edges = []
        for edge in edges:
            if edge.is_self_loop():
                edge.a.superiors.remove(edge.b)
                del_edges.append(edge)

        for del_edge in del_edges:
            edges.remove(del_edge)

        return edges

    def __remove_transitivity(self, nodes: list[Node], edges: list[Edge]) -> list[Edge]:
        for node in nodes:
            for superior in node.superiors:
                edges = self.__dfs_search(node, superior, edges, [])

        return edges

    def __hasse_process(self, nodes: list[Node], edges: list[Edge]) -> list[Edge]:
        edges = self.__remove_self_loops(edges)
        edges = self.__remove_transitivity(nodes, edges)

        return edges