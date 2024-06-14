import numpy as np

from helpers import get_relation, get_relation_score

def kendall_distance(x, y):
    shape = x.shape
    assert shape == y.shape
    if shape == None:
        raise ValueError("Matrices must be an numpy 2d array")
    distance = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i == j:
                continue
            rel_x = get_relation(x[i][j], x[j][i])
            rel_y = get_relation(y[i][j], y[j][i])
            distance += get_relation_score(rel_x, rel_y)
    
    distance *= 1/8
    return distance

def kendall_tau(distance, n):
    kendall_coef = 1 - 4 * distance / (n * (n - 1))
    return kendall_coef

def max_rank_diff(size: int):
    if size % 2 == 0:
        return (size // 2) * size
    else:
        return -(size // -2) * (size - 1)

def normalized_hit_ratio(r1, r2):
    common_leaders = len(np.intersect1d(r1.leaders, r2.leaders))
    all_leaders = len(np.union1d(r1.leaders, r2.leaders))

    return common_leaders / all_leaders
