from graphviz import Digraph
from .hasse import Hasse
from ...types import QuadraticArrayType

class Graph(Hasse):
    def __init__(self, matrix: QuadraticArrayType, labels: list[str], *options: dict):
        super().__init__(matrix, labels)

        try:
            self.G = Digraph(format='png', strict=True, *options)
        except Exception as e:
            raise e

        self.__initialize_structure()
        self.G.attr("node", shape="box")

    def __initialize_structure(self):
        for node in self.nodes:
            self.G.node(str(node))

        for edge in self.edges:
            self.G.edge(str(edge.b), str(edge.a))

    def save(self, filename: str, view: bool=False):
        self.G.render(filename, view=view)

    def show(self, filename: str = "digraph", directory: str = "./graphs/"):
        self.G.view(filename=filename, directory=directory)
    
    def get_graph(self) -> Digraph:
        return self.G
