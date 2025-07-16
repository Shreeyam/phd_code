# cameras.py
import numpy as np
from dynamic_tasker.orbits import *
from dynamic_tasker.rotations import *
from dynamic_tasker.vector import *
# Origin at bottom left corner
def get_intrinsics(f, c_x, c_y):
    K = np.hstack([np.array([-f, 0, c_x, 0, f, c_y, 0, 0, 1]).reshape(3,3), np.zeros((3,1))])
    return K

def get_intrinsics_from_fov(fov, width, height, axis='x'):
    if axis == 'x':
        f = width / (2 * np.tan(np.deg2rad(fov / 2)))
    elif axis == 'y':
        f = height / (2 * np.tan(np.deg2rad(fov / 2)))
    else:
        raise ValueError(f"Invalid axis: {axis}")
        
    return get_intrinsics(f, width/2, height/2)

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

def unproject(P, img_points, depth):
    """
    Unprojects 2D image points to 3D world coordinates given their depth values.
    
    Parameters
    ----------
    P : np.ndarray
        The 3x4 camera projection matrix (assumed to be of the form K[R | -R*t]).
    img_points : np.ndarray
        An (N x 2) array of image (pixel) coordinates.
    depth : float or np.ndarray
        The depth value(s) corresponding to the camera z-coordinate (before homogeneous division).
        This can be a single scalar (applied to all points) or an (N,) array.
    
    Returns
    -------
    world_points : np.ndarray
        An (N x 3) array of unprojected 3D world coordinates.
    
    Notes
    -----
    Because the projection operation loses the depth information, unprojection is ambiguous 
    unless the depth is provided. This function inverts the operation:
    
        [u, v]  -->  d*[u, v, 1]^T = M*X + p4  with M = P[:, :3] and p4 = P[:, 3],
    
    and returns:
    
        X = inv(M) * (d*[u, v, 1]^T - p4)
    """
    # Ensure depth is an (N, 1) array.
    if np.isscalar(depth):
        depth = np.full((img_points.shape[0], 1), depth)
    else:
        depth = np.array(depth).reshape(-1, 1)
    
    # Decompose P into its 3x3 part and the translation part.
    M = P[:, :3]    # should be invertible (e.g. K*R)
    p4 = P[:, 3]    # shape (3,)
    
    # Build the homogeneous image coordinates scaled by the depth.
    # For each 2D point [u, v], we form [u*d, v*d, d]
    homogeneous_img = np.hstack([img_points * depth, depth])
    
    # Solve for the world point: X = inv(M) @ (d*[u,v,1] - p4)
    invM = np.linalg.inv(M)
    # Subtract p4 from each row (broadcasting works because p4 is (3,))
    world_points = (homogeneous_img - p4) @ invM.T
    
    return world_points

def project_from_orbit(points, K, orbit, time, roll_angle=0, pitch_angle=15):
    r, v = kepler2eci(propagate_orbit(orbit, time))
    # Figure out the quaternion pointing to nadir
    v_unit = v / np.linalg.norm(v)
    r_unit = r / np.linalg.norm(r)

    R_t = rotmat_from_vec(r_unit, -v_unit)
    R_q = eul2R(0, np.deg2rad(roll_angle), np.deg2rad(pitch_angle))
    R_t = R_t @ R_q
    R_t = np.linalg.inv(R_t)
    R_t = R_t[[2, 1, 0], :]
    P = get_camera_matrix(K, R_t, r)

    # Check if points is not empty
    if(points.size > 0):
        projected_points = project(P, points, False)
        return projected_points
    else:
        return []

def unproject_from_orbit(img_points, depth, K, orbit, time, roll_angle=0, pitch_angle=15):
    """
    Unprojects 2D image points to 3D world coordinates using orbit information.
    
    Parameters
    ----------
    img_points : np.ndarray
        An (N x 2) array of image (pixel) coordinates.
    depth : float or np.ndarray
        The depth value(s) corresponding to the camera z-coordinate (before homogeneous division).
        Can be a scalar (applied to all points) or an array of shape (N,).
    K : np.ndarray
        The 3x3 intrinsic calibration matrix.
    orbit : object
        The orbit parameters (or object) used by propagate_orbit and kepler2eci.
    time : float
        The time at which to propagate the orbit.
    roll_angle : float, optional
        The camera roll angle in degrees (default is 0).
    pitch_angle : float, optional
        The camera pitch angle in degrees (default is 15).

    Returns
    -------
    world_points : np.ndarray
        An (N x 3) array of 3D world coordinates corresponding to the unprojected image points.
    
    Notes
    -----
    This function first computes the camera position and orientation from the orbit parameters:
    
        - Propagate the orbit to the given time and convert to ECI coordinates.
        - Compute a rotation matrix pointing toward nadir using the unit vectors of the 
          position (r_unit) and velocity (v_unit).
        - Adjust the rotation using the specified roll and pitch.
        - Compute the camera projection matrix P = get_camera_matrix(K, R_t, r).
    
    Then it uses the provided depth to unproject the 2D image points back into 3D space via:
    
        X = inv(M) @ (d*[u, v, 1]^T - p4)
    
    where M = P[:, :3] and p4 = P[:, 3].
    
    Dependencies
    ------------
    This function assumes that the following functions are defined elsewhere:
        - propagate_orbit(orbit, time)
        - kepler2eci(state)
        - rotmat_from_vec(target_vec, up_vec)
        - eul2R(yaw, pitch, roll)
        - get_camera_matrix(K, R, t)
        - unproject(P, img_points, depth)
    """
    # Propagate the orbit to the given time and convert to ECI coordinates.
    r, v = kepler2eci(propagate_orbit(orbit, time))
    
    # Compute unit vectors from position and velocity.
    r_unit = r / np.linalg.norm(r)
    v_unit = v / np.linalg.norm(v)
    
    # Compute the rotation matrix that aligns the camera with nadir (pointing toward Earth).
    R_t = rotmat_from_vec(r_unit, -v_unit)
    # Adjust for roll and pitch (convert angles from degrees to radians).
    R_q = eul2R(0, np.deg2rad(roll_angle), np.deg2rad(pitch_angle))
    R_t = R_t @ R_q
    # Invert the rotation to get the camera-to-world rotation.
    R_t = np.linalg.inv(R_t)
    # Reorder axes as in the project_from_orbit function.
    R_t = R_t[[2, 1, 0], :]
    
    # Construct the full camera projection matrix.
    P = get_camera_matrix(K, R_t, r)
    
    # Unproject the image points using the provided depth.
    world_points = unproject(P, img_points, depth)
    
    return world_points

def ecef2pitchroll(pos_ecef, v_ecef, vec):
    Up = pos_ecef / np.linalg.norm(pos_ecef)

    Along = v_ecef / np.linalg.norm(v_ecef)
    Right = np.cross(Along, Up)
    Right = Right / np.linalg.norm(Right)

    Along = np.cross(Up, Right)
    Along = Along / np.linalg.norm(Along)

    R_ecef_to_body = np.vstack([Right, Along, Up])

    v_local = R_ecef_to_body @ vec
    v_local_norm = v_local / np.linalg.norm(v_local)  # normalize for angle calculations

    pitch = -np.arctan2(v_local_norm[1], v_local_norm[2])
    roll  = -np.arctan2(v_local_norm[0], v_local_norm[2])

    pitch_deg = np.degrees(pitch) * 1
    roll_deg  = np.degrees(roll)

    return pitch_deg, roll_deg

def project_in_box(pitch_deg, roll_deg, orbit, t, accesses, points, width, height, K):
    # First, project and see if it's in the box
    points_eci = np.array([ecef2eci(p, t) for p in points])
    ecef_projected_dir = project_from_orbit(points_eci, K, orbit, t, pitch_angle=pitch_deg, roll_angle=roll_deg)
    # Figure out how many are in the box
    in_box_idx = np.array([i for i, p in enumerate(ecef_projected_dir) if p[0] >= 0 and p[0] <= width and p[1] >= 0 and p[1] <= height])
    return [a for i, a in enumerate(accesses) if i in in_box_idx], in_box_idx, ecef_projected_dir

def filter_accesses_horizon(orbit, time, accesses, pos_ecef, field_of_regard=30):
    return [(r, a, t, access, idx) for r, a, t, access, idx in accesses if t >= time and t <= time + datetime.timedelta(seconds=horizon_time(orbit)) and a <= field_of_regard and a >= -field_of_regard]

def create_box(width, height, points_per_edge=0):
    # Create the base box corners
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    
    # Create high resolution edges
    edges = []
    for i in range(4):
        start = corners[i]
        end = corners[(i + 1) % 4]
        # Generate evenly spaced points along each edge
        edge_points = np.linspace(start, end, points_per_edge)
        edges.append(edge_points[:-1])  # Exclude last point to avoid duplicates
    
    # Combine all edges and add the closing point
    high_res_box = np.vstack(edges + [corners[0]])
    return high_res_box
    