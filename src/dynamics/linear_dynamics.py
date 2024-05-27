import numpy as np
from quat.quat_helpers import *

# 8 dimensional state vector
def xdot(x, u):
    w = x[0:3]
    q = x[3:]

    xdot[0:3] = u
    xdot[3:] = 0.5 * qdot_matrix(x[3:]) @ w

    return xdot