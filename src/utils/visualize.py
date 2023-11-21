from pyvis.network import Network
from itertools import product

def get_edges(matrix):
    shape = matrix.shape[0]
    con = [(i, j) for i, j in product(range(shape), range(shape)) if i != j and matrix[i][j] > matrix[j][i]]
    return con


def visualize_ranking(matrix):
    net = Network(directed=True)
    
    net.add_nodes(range(matrix.shape[0]), label=[str(i) for i in range(matrix.shape[0])])
    net.add_edges(get_edges(matrix))

    net.show("graph.html", notebook=False)