from graphviz import Digraph
from .hasse import Hasse

class Graph(Hasse):
    def __init__(self, matrix, labels):
        super().__init__(matrix, labels)
        self.G = Digraph(format='png', strict=True)
        self.G.attr("node", shape="box")
        
        self.draw()

    def draw(self):
        for node in self.nodes:
            self.G.node(str(node))
        for edge in self.edges:
            self.G.edge(str(edge[1]), str(edge[0]))
        
        self.G.render("graph", view=True)
