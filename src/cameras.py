import numpy as np
from rotations import *

def get_intrinsics(f, c_x, c_y):
    K = np.hstack([np.array([-f, 0, c_x, 0, -f, c_y, 0, 0, 1]).reshape(3,3), np.zeros((3,1))])
    return K

def get_extrinsics(q, p):
    R_q = q2mat(q).float()
    R_0 = eul2R(0, np.pi, -np.pi/2).float() # boresight on -z
    R = R_q @ R_0
    t = -R @ p
    bottom_row = np.tensor([[0, 0, 0, 1]])
    Rt = np.concatenate((R, t.reshape(3, 1)), dim=1)
    transformation_matrix = np.concatenate((Rt, bottom_row), dim=0)
    return transformation_matrix
