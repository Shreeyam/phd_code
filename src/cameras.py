import numpy as np
from rotations import *

def get_intrinsics(f, c_x, c_y):
    K = np.hstack([np.array([f, 0, c_x, 0, f, c_y, 0, 0, 1]).reshape(3,3), np.zeros((3,1))])
    return K

def get_extrinsics(R_t, p):
    R_q = R_t #q2mat(q)
    R_0 = np.eye(3) # eul2R(-np.pi/4, np.pi/2, 0) # boresight on -z
    R = R_q @ R_0
    t = -R @ p
    bottom_row = np.array([[0, 0, 0, 1]])
    Rt = np.hstack([R, t.reshape(3, 1)])
    transformation_matrix = np.vstack([Rt, bottom_row])
    return transformation_matrix

def get_camera_matrix(K, R, p):
    return K @ get_extrinsics(R, p)

def project(P, points, z_clip=True):
    # Project points to image plane
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    points = (P @ points.T).T

    # Clip all points with z < 0
    if(z_clip):
        points = points[points[:, 2] > 0]

    points = points / points[:, [2]]
    points = points[:, 0:2]

    return points

# todo: review this function...
def unproject(P, points):
    # Unproject points to 3D space
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    points = points @ np.linalg.inv(P.T)
    points = points / points[:, 3:]
    points = points[:, :3]

    return points