# rotations.py
import numpy as np

def qconj(q):
    p = q.copy()
    p[1:] *= -1
    return p

def qmult(q1, q2):
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    # Use geometric product to multiply quaternions
    w = w1*w2 - np.dot(v1, v2)
    v = w1*v2 + w2*v1 + np.cross(v1, v2)
    return np.hstack([w, v])

def qsandwich(q, v):
    p = np.hstack([0, v])
    return qmult(q, qmult(p, qconj(q)))[1:]

def qarray(q, func, *args):
    return np.array([func(qi, *args) for qi in q])

def q2mat(q, homogenous=False):
    w, x, y, z = q
    if(not homogenous):
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])
    
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w, 0],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w, 0],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2, 0],
        [0, 0, 0, 1]
    ])

def eul2q(yaw, pitch, roll):
    # Calculate the half angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Calculate the quaternion
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)

def rotmat_x(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotmat_y(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotmat_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def eul2R(roll, pitch, yaw):
    
    # Calculate rotation matrix for roll (around x-axis)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    # Calculate rotation matrix for pitch (around y-axis)
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    # Calculate rotation matrix for yaw (around z-axis)
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Combine the rotations
    R = R_z @ (R_y @ R_x)
    
    return R

def rotmat_from_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes the rotation matrix that rotates the standard basis vectors
    x = [1, 0, 0] and y = [0, 1, 0] to the provided orthogonal vectors a and b.

    Parameters:
    a (np.ndarray): The target vector for the x-axis (should be a 3-element array).
    b (np.ndarray): The target vector for the y-axis (should be a 3-element array).

    Returns:
    np.ndarray: A 3x3 rotation matrix.
    
    Raises:
    ValueError: If the input vectors are not 3-dimensional or not orthogonal.
    """
    # Ensure input vectors are numpy arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    # Check if vectors are 3-dimensional
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Input vectors must be three-dimensional.")
    
    # Normalize the vectors
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    if a_norm == 0 or b_norm == 0:
        raise ValueError("Input vectors must be non-zero.")
    
    a_unit = a / a_norm
    b_unit = b / b_norm
    
    # Check orthogonality
    dot_product = np.dot(a_unit, b_unit)
    if not np.isclose(dot_product, 0.0, atol=1e-8):
        raise ValueError("Input vectors must be orthogonal.")
    
    # Compute the third orthogonal vector using cross product
    c_unit = np.cross(a_unit, b_unit)
    
    # Form the rotation matrix with a_unit, b_unit, c_unit as columns
    R = np.column_stack((a_unit, b_unit, c_unit))
    
    # Verify that R is a valid rotation matrix
    if not np.allclose(np.dot(R, R.T), np.identity(3), atol=1e-8):
        raise ValueError("Resulting matrix is not orthogonal.")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-8):
        raise ValueError("Resulting matrix does not have a determinant of +1.")
    
    return R