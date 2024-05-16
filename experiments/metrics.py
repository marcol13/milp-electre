import numpy as np

def kendall_distance(x, y):
    shape = x.shape
    assert shape == y.shape
    if shape == None:
        return 1.0
    x = np.asarray(x)
    y = np.asarray(y)
    distance = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            distance += abs(x[i, j] - y[i, j])
    
    distance *= 1/2
    return distance

def kendall_tau(distance, n):
    kendall_coef = 1 - 4 * distance / (n * (n - 1))
    return kendall_coef

def max_rank_diff(size: int):
    if size % 2 == 0:
        return (size // 2) * size
    else:
        return -(size // -2) * (size - 1)
