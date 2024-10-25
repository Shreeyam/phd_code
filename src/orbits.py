import numpy as np
from constants import Constants
from collections import namedtuple
from rotations import *

def lat2long2ecef(latlong):
    lat_deg, lon_deg = latlong
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    X = Constants.R_E * np.cos(lat_rad) * np.cos(lon_rad)
    Y = Constants.R_E * np.cos(lat_rad) * np.sin(lon_rad)
    Z = Constants.R_E * np.sin(lat_rad)

    return np.array([X, Y, Z])

def ecef2eci(ecef, time):
    time_difference = time - Constants.J2000
    days_difference = time_difference.total_seconds() / (24 * 60 * 60)
    theta = np.deg2rad((Constants.ERA_J2000 + Constants.gamma * days_difference) % 360)
    rot_mat = rotmat_z(theta)
    return np.dot(rot_mat, ecef)

def eci2ecef(eci, time):
    time_difference = time - Constants.J2000
    days_difference = time_difference.total_seconds() / (24 * 60 * 60)
    theta = np.deg2rad((Constants.ERA_J2000 + Constants.gamma * days_difference) % 360)
    rot_mat = rotmat_z(-theta)
    return np.dot(rot_mat, eci)

def ecef2latlong(ecef):
    X, Y, Z = ecef
    lat = np.arcsin(Z / np.linalg.norm(ecef))
    lon = np.arctan2(Y, X)
    return np.array([np.rad2deg(lat), np.rad2deg(lon)])

def ecef2eci_vec(ecef, time):
    # Ensure ecef is a NumPy array
    ecef = np.asarray(ecef)
    
    # Check the shape of ecef
    if ecef.ndim != 2 or ecef.shape[1] != 3:
        raise ValueError("ecef must be a 2D array with shape (N, 3)")
    
    time_difference = time - Constants.J2000
    days_difference = time_difference.total_seconds() / (24 * 60 * 60)
    
    # Calculate the rotation angle theta in radians
    theta_deg = (Constants.ERA_J2000 + Constants.gamma * days_difference) % 360
    theta = np.deg2rad(theta_deg)
    
    # Compute the rotation matrix
    rot_mat = rotmat_z(theta)  # Shape: (3, 3)
    
    # Apply the rotation matrix to all ECEF coordinates
    # Using matrix multiplication: (N, 3) @ (3, 3).T -> (N, 3)
    eci = ecef @ rot_mat.T
    
    return eci

def latlong2ecef_vec(latlong):
    lat_deg = latlong[:, 0]
    lon_deg = latlong[:, 1]
    
    # Convert degrees to radians
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    
    # Compute ECEF coordinates
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)
    
    X = Constants.R_E * cos_lat * cos_lon
    Y = Constants.R_E * cos_lat * sin_lon
    Z = Constants.R_E * sin_lat
    
    # Stack into a (N, 3) array
    ecef = np.column_stack((X, Y, Z))
    
    return ecef

# Calculate the vector pointing to the sun in the ECI frame
def sunvec_eci(time):
    time_difference = time - Constants.J2000
    d = time_difference.total_seconds() / (24 * 60 * 60) 
    L = 280.4606184 + [(36000.77005361 / 36525) * d] # mean longitude
    g = 357.5277233 + ([35999.05034 / 36525] * d) # mean anomaly
    p = L + [1.914666471 * np.sin(g * np.pi / 180)] + [0.918994643 * np.sin(2*g * np.pi / 180)] # ecliptic longitude lambda
    q = 23.43929 - ((46.8093/3600) * (d / 36525)) # obliquity of the ecliptic plane epsilon
    return np.array([np.cos(p * np.pi / 180), np.cos(q * np.pi / 180) * np.sin(p * np.pi / 180), np.sin(q * np.pi / 180) * np.sin(p * np.pi / 180)])


# Define a namedtuple for Keplerian elements
Keplerian = namedtuple('Keplerian', ['a', 'e', 'i', 'omega', 'Omega', 'M'])

# Six keplerian elements: a, e, i, omega, Omega, M
def kepler2eci(elements: tuple) -> tuple:
    a, e, i, omega, Omega, M = elements
    mu = Constants.mu
    nu = None
    r = None
    if(e != 0):
        # Solve Kepler's Equation for E (Eccentric Anomaly)
        E = M
        error = 1
        while error > 1e-6:
            E_new = M + e * np.sin(E)
            error = np.abs(E_new - E)
            E = E_new
    
        # True anomaly
        nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        # Distance (radius) from the central body
        r = a * (1 - e * np.cos(E))
    else:
        nu = M
        r = a

    # Position in the perifocal coordinate system
    r_perifocal = np.array([r * np.cos(nu), r * np.sin(nu), 0])
    
    # Velocity in the perifocal coordinate system
    h = np.sqrt(mu * a * (1 - e**2))
    v_perifocal = np.array([-mu / h * np.sin(nu), mu / h * (e + np.cos(nu)), 0])

    # Rotation matrices
    R_Omega = np.array([
        [np.cos(Omega), -np.sin(Omega), 0],
        [np.sin(Omega), np.cos(Omega), 0],
        [0, 0, 1]
    ])
    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    R_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])

    # Complete rotation matrix from perifocal to ECI
    R = R_Omega @ R_i @ R_omega

    # Position and velocity in the ECI frame
    r_eci = R @ r_perifocal
    v_eci = R @ v_perifocal

    return r_eci, v_eci

# Generate a circular orbit
def circular_orbit(a: float, i: float, Omega: float, M: float):
    return Keplerian(a, 0, i, 0, Omega, M)

def propagate_orbit(orbit: Keplerian, time: float):
    mu = Constants.mu
    a, e, i, omega, Omega, M = orbit
    n = np.sqrt(mu / a**3)
    M_new = M + n * time
    return Keplerian(a, e, i, omega, Omega, M_new)

def v_orb(h):
    return np.sqrt(Constants.mu / (h + Constants.R_E))