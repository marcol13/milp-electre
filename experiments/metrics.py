import numpy as np

def kendall_tau(x, y):
    shape = x.shape
    assert shape == y.shape
    if shape == None:
        return 1.0
    x = np.asarray(x)
    y = np.asarray(y)
    dis = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            dis += abs(x[i, j] - y[i, j])
    
    dis *= 1/2
    kendall_coef = 1 - 4 * dis / (shape[0] * (shape[0] - 1))
    return kendall_coef

