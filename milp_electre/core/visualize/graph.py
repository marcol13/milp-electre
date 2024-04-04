from pyvis.network import Network
from .hasse import Hasse

class Graph(Hasse):
    def __init__(self, matrix, labels):
        super().__init__(matrix, labels)
        self.net = Network(directed=True, heading="Graph")
        
        self.draw()

    def draw(self):
        for node in self.nodes:
            y = int((self.max_level - node.level) * 50)
            self.net.add_node(node.name, shape="box", level=node.level, y=y, physics=False)
        for edge in self.edges:
            self.net.add_edge(edge[1].name, edge[0].name)
        
        self.net.toggle_physics(False)
        self.net.show("index.html", notebook=False)

        # X = nx.nx_agraph.from_agraph(A)
