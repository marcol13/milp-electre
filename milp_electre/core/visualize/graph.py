from graphviz import Digraph
from .hasse import Hasse
from ..types import QuadraticArrayType

class Graph(Hasse):
    #TODO: Add possibility to pass options data for graphviz rendering
    def __init__(self, matrix: QuadraticArrayType, labels: list[str]):
        super().__init__(matrix, labels)
        self.G = Digraph(format='png', strict=True)

        for node in self.nodes:
            self.G.node(str(node))

        for edge in self.edges:
            self.G.edge(str(edge[1]), str(edge[0]))

        self.G.attr("node", shape="box")

    def save(self, filename: str, view: bool=False):
        self.G.render(filename, view=view)

    def show(self):
        self.G.view()
    
    def get_graph(self) -> Digraph:
        return self.G
