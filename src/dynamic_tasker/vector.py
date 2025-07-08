import numpy as np

def dist(x, y):
    if(len(x.shape) == 1 and len(y.shape) == 1):
        return np.linalg.norm(x - y)

    return np.linalg.norm(x - y, axis=1)

def dist2plane(r, n, p):
    return ((p - r).dot(n) / np.linalg.norm(n))