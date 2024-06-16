import numpy as np

from experiments.helpers import get_relation, get_relation_score
from mcdalp.core.relations import PositivePreference, NegativePreference, Indifference, Incomparible


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
    
def get_preference(i: int, j: int):
        if i > j:
            return PositivePreference
        elif j > i:
            return NegativePreference
        elif i == j == 1:
            return Indifference
        else:
            return Incomparible
    
def rdm(r1, r2, mode="complete"):
    assert r1.outranking.shape == r2.outranking.shape
    # assert r1.scores == r2.scores

    shape = r1.outranking.shape[0]

    denom = 4 * shape * (shape - 1)
    measure= 0
    for i in range(shape):
        for j in range(shape):
            if i == j:
                continue
            r1_pref = get_preference(r1.outranking[i][j], r1.outranking[j][i])
            r2_pref = get_preference(r2.outranking[i][j], r2.outranking[j][i])

            measure += r1.scores.score_matrix[r1_pref][r2_pref]

    return measure / denom

def normalized_hit_ratio(r1, r2):
    common_leaders = len(np.intersect1d(r1.leaders, r2.leaders))
    all_leaders = len(np.union1d(r1.leaders, r2.leaders))

    return common_leaders / all_leaders

def second_kendall(r1, r2):
    rdm_score = rdm(r1, r2, "partial")
    return 1 - 2 * rdm_score