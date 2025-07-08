# orbits.py
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
    Intersect (or tangent-snap) a ray P + t u with a sphere of centre x0, radius r.

    Returns
    -------
    (pt1, pt2, t1, t2)
        • Two 3-D points (may be the same) and their parameter values along the ray.
        • If the ray does not meet the sphere and horizon_snap is False → None.
        • If horizon_snap is True and there is no intersection → the tangent
          point is returned twice, with the appropriate single parameter t.
    """
    P   = np.asarray(P, dtype=float)
    u   = np.asarray(u, dtype=float)
    x0  = np.asarray(x0, dtype=float)
    if np.allclose(u, 0):
        raise ValueError("Direction vector u must be non-zero")

    # Normalise the direction so that 't' is in metres (or whatever units you use)
    u   = u / np.linalg.norm(u)

    # Quadratic coefficients for |P + t u – x0|² = r²
    d   = P - x0
    A   = 1.0                            # because u is unit
    B   = 2.0 * np.dot(u, d)
    C   = np.dot(d, d) - r*r
    disc = B*B - 4*A*C

    # ---------------- regular intersection cases ----------------
    if disc > 0:                         # two points
        sqrt_disc = np.sqrt(disc)
        t1 = (-B + sqrt_disc) / (2*A)
        t2 = (-B - sqrt_disc) / (2*A)
        return (P + t1*u, P + t2*u, t1, t2)

    if np.isclose(disc, 0):              # exactly tangent already
        t = -B / (2*A)
        pt = P + t*u
        return (pt, pt, t, t)

    # ---------------- no hit, maybe horizon-snap? ----------------
    if not horizon_snap:
        return None

    # ---- compute tangent point in the plane spanned by d and u ----
    L2 = np.dot(d, d)
    if L2 <= r*r:
        # P happens to lie on or inside the sphere – snapping undefined
        return None

    # Plane normal and an in-plane unit vector perpendicular to d
    n = np.cross(u, d)
    if np.linalg.norm(n) < 1e-12:        # u ‖ d  ⇒  choose any perpendicular axis
        n = np.cross(d, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(n) < 1e-12:    # unlucky – d happens to be x-axis
            n = np.cross(d, np.array([0.0, 1.0, 0.0]))
    k = np.cross(n, d)
    k_hat = k / np.linalg.norm(k)

    L    = np.sqrt(L2)
    delta = np.sqrt(L2 - r*r)

    # Two candidate tangent points
    T1 = x0 + (r*r / L2) * d + (r * delta / L) * k_hat
    T2 = x0 + (r*r / L2) * d - (r * delta / L) * k_hat

    # Choose the one that lies "forward" along u (positive dot)
    dir1 = (T1 - P)
    dir2 = (T2 - P)
    dot1 = -np.dot(dir1, u)
    dot2 = -np.dot(dir2, u)

    if dot1 >= 0 and dot2 < 0:
        T = T1; t = -dot1                 # u is unit, so t = proj length
    elif dot2 >= 0 and dot1 < 0:
        T = T2; t = -dot2
    else:
        # Both are forward (rare) – pick the smaller angular offset
        ang1 = np.arccos(dot1 / np.linalg.norm(dir1))
        ang2 = np.arccos(dot2 / np.linalg.norm(dir2))
        T, t = (T1, dot1) if ang1 < ang2 else (T2, dot2)

    return (T, T, t, t)
    
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