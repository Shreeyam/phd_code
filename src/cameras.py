import numpy as np
from orbits import *
from rotations import *

# Origin at bottom left corner
def get_intrinsics(f, c_x, c_y):
    K = np.hstack([np.array([-f, 0, c_x, 0, f, c_y, 0, 0, 1]).reshape(3,3), np.zeros((3,1))])
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

def project_from_orbit(points, orbit, time, roll_angle=0):
    width = 800
    height = 600
    r, v = kepler2eci(propagate_orbit(orbit, time))
    K = get_intrinsics(300, width//2, height//2)
    # Figure out the quaternion pointing to nadir
    v_unit = v / np.linalg.norm(v)
    r_unit = r / np.linalg.norm(r)

    R_t = rotation_matrix(r_unit, -v_unit)
    R_q = eul2R(0, np.deg2rad(roll_angle), np.deg2rad(15))
    R_t = R_t @ R_q
    R_t = np.linalg.inv(R_t)
    R_t = R_t[[2, 1, 0], :]
    P = get_camera_matrix(K, R_t, r)
    projected_points = project(P, points, False)
    # projected_points[:, 0] = width - projected_points[:, 0]
    return projected_points