import numpy as np
import datetime
from collections import namedtuple
from typing import Union

from dynamic_tasker.constants import Constants
from dynamic_tasker.rotations import *

def latlong2ecef(latlong):
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

def latlong2eci(lat, long, time):    
    # Convert latlong to ECEF coordinates
    ecef = latlong2ecef((lat, long))
    
    # Convert ECEF to ECI coordinates
    eci = ecef2eci(ecef, time)
    
    return eci

# Calculate the vector pointing to the sun in the ECI frame
def sunvec_eci(time):
    time_difference = time - Constants.J2000
    d = time_difference.total_seconds() / (24 * 60 * 60) 
    L = 280.4606184 + ((36000.77005361 / 36525) * d) # mean longitude
    g = 357.5277233 + ((35999.05034 / 36525) * d) # mean anomaly
    p = L + (1.914666471 * np.sin(g * np.pi / 180)) + (0.918994643 * np.sin(2*g * np.pi / 180)) # ecliptic longitude lambda
    q = 23.43929 - ((46.8093/3600) * (d / 36525)) # obliquity of the ecliptic plane epsilon
    return np.array((np.cos(p * np.pi / 180), np.cos(q * np.pi / 180) * np.sin(p * np.pi / 180), np.sin(q * np.pi / 180) * np.sin(p * np.pi / 180)))


# Define a namedtuple for Keplerian elements
Keplerian = namedtuple('Keplerian', ['a', 'e', 'i', 'omega', 'Omega', 'M', 't'])

# Six keplerian elements: a, e, i, omega, Omega, M
def kepler2eci(elements: Keplerian) -> tuple:
    a, e, i, omega, Omega, M, t = elements
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
def circular_orbit(a: float, i: float, Omega: float, M: float, t: datetime) -> Keplerian:
    return Keplerian(a, 0, i, 0, Omega, M, t)

def propagate_orbit(orbit: Keplerian, time: Union[float, datetime.datetime, datetime.timedelta]) -> Keplerian:
    mu = Constants.mu
    a, e, i, omega, Omega, M, t = orbit
    n = np.sqrt(mu / a**3)
    if(isinstance(time, datetime.timedelta)):
        time = time.total_seconds()
    elif(isinstance(time, datetime.datetime)):
        time = (time - t).total_seconds()
    
    M_new = M + n * time
    return Keplerian(a, e, i, omega, Omega, M_new, t + datetime.timedelta(seconds=time))

def v_orb(h):
    return np.sqrt(Constants.mu / (h + Constants.R_E))

def t_orb(elements: Keplerian):
    return 2 * np.pi * np.sqrt((elements.a**3) / Constants.mu)

def horizon_distance(elements):
    return np.sqrt((elements.a)**2 - Constants.R_E**2)

def horizon_angle(elements):
    return np.arcsin(Constants.R_E / elements.a)

def horizon_time(elements):
    return horizon_distance(elements) / v_orb(elements.a)

def horizon_spherical_angle(elements):
    return np.arccos(Constants.R_E / elements.a)

def intersect_ray_sphere(P, u, x0, r, horizon_snap=False):
    """
    Determines the intersections of a ray with a sphere.
    
    Parameters:
    P (numpy array): The starting point of the ray (3D vector).
    u (numpy array): The direction of the ray (3D vector).
    x0 (numpy array): The center of the sphere (3D vector).
    r (float): The radius of the sphere.
    horizon_snap (bool): If True, the ray will be snapped to the horizon if it does not intersect the sphere.

    Returns:
    t1, t2 (float, float): The parameter values at which the intersections occur.
    None if there are no intersections.
    """
    # Normalize direction vector
    u = u / np.linalg.norm(u)
    
    # Compute coefficients of the quadratic equation
    A = np.dot(u, u)
    B = 2 * np.dot(u, P - x0)
    C = np.dot(P - x0, P - x0) - r**2
    
    # Compute the discriminant
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        # No intersection
        if(horizon_snap):
            # Find the closest point on the ray to the sphere center
            t_closest = -np.dot(P - x0, u)
            closest_point = P + t_closest * u
            
            # Project this point onto the sphere surface (horizon point)
            direction_to_surface = closest_point - x0
            direction_normalized = direction_to_surface / np.linalg.norm(direction_to_surface)
            horizon_point = x0 + r * direction_normalized
            
            # Return as 4-tuple with same point twice and the calculated t value
            return (horizon_point, horizon_point, t_closest, t_closest)
        else:
            return None
    elif discriminant == 0:
        # One intersection (tangent)
        t = -B / (2*A)
        point = P + t * u
        return (point, point, t, t)
    else:
        # Two intersections
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-B + sqrt_disc) / (2*A)
        t2 = (-B - sqrt_disc) / (2*A)
        return (P + t1 * u, P + t2 * u, t1, t2)
    
def earth_line_intersection(P, u, horizon_snap=False):
    p = intersect_ray_sphere(P, u, np.array([0, 0, 0]), Constants.R_E, horizon_snap)
    if(p is None):
        return p
    else:
        p1, p2, t1, t2 = p
        if(t1 < 0 and t2 < 0):
            return (p1, p2)
        else:
            return None
        
def split_orbit_track(latlongs, threshold=180):
    lat, long = np.array(latlongs)[:, 0], np.array(latlongs)[:, 1]
    # Calculate the difference between consecutive longitudes
    delta_long = np.abs(np.diff(long))
    
    # Identify where the jump exceeds the threshold
    jump_indices = np.where(delta_long > threshold)[0] + 1
    
    # Split the data at the jump indices
    segments = np.split(latlongs, jump_indices)
    return segments

def kepler2latlong(orbit: Keplerian, time):
    return ecef2latlong(eci2ecef(kepler2eci(propagate_orbit(orbit, (time - orbit.t).total_seconds()))[0], time))