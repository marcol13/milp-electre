from pyvis.network import Network

def visualize_ranking(matrix):
    net = Network()
    net.add_node(1, label="Node 1") # node id = 1 and label = Node 1
    net.add_node(2) # node id and label = 2
    net.add_edge(1,2)
    net.show()