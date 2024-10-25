import numpy as np

def dist(x, y):
    return np.linalg.norm(x - y, axis=1)

def dist2plane(r, n, p):
    return ((p - r).dot(n) / np.linalg.norm(n))