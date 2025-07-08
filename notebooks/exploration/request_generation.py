import numpy as np
import datetime
import matplotlib.pyplot as plt

R_E = 6378 # km
# J2000 epoch
J2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
ERA_J2000 = 280.46
gamma=360.9856123035484

def latlong2ecef(latlong):
    lat_deg, lon_deg = latlong
    lat_rad = np.rad2deg(lat_deg)
    lon_rad = np.rad2deg(lon_deg)

    X = R_E * np.cos(lat_rad) * np.cos(lon_rad)
    Y = R_E * np.cos(lat_rad) * np.sin(lon_rad)
    Z = R_E * np.sin(lat_rad)

    return np.array([X, Y, Z])

def rotmat_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def ecef2eci(ecef, time):
    time_difference = time - J2000
    days_difference = time_difference.total_seconds() / (24 * 60 * 60)
    theta = np.deg2rad((ERA_J2000 + gamma * days_difference) % 360)
    rot_mat = rotmat_z(theta)
    return np.dot(rot_mat, ecef)

# Generate tasks in lat/long coordinates
task = np.array([0, 51.6]) # Latitude, Longitude
task_ecef = latlong2ecef(task)
task_eci = ecef2eci(task_ecef, datetime.datetime.now())
print(task_eci)

from collections import namedtuple

# Define a namedtuple for Keplerian elements
Keplerian = namedtuple('Keplerian', ['a', 'e', 'i', 'omega', 'Omega', 'M'])

# Six keplerian elements: a, e, i, omega, Omega, M
def kepler2eci(elements: Keplerian) -> np.array:
    a, e, i, omega, Omega, M = elements
    # Constants
    # n = np.sqrt(mu / a**3) mean motion, not needed
    r = a
    if(e==0):
        nu = M
    else:
        E = M
        error = 1
        while error > 1e-6:
            E_new = M + e * np.sin(E)
            error = np.abs(E_new - E)
            E = E_new
            nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
            r = a * (1 - e * np.cos(E))

    x = r * (np.cos(Omega) * np.cos(omega + nu) - np.sin(Omega) * np.sin(omega + nu) * np.cos(i))
    y = r * (np.sin(Omega) * np.cos(omega + nu) + np.cos(Omega) * np.sin(omega + nu) * np.cos(i))
    z = r * np.sin(omega + nu) * np.sin(i)
    return np.array([x, y, z])

# Generate a circular orbit
def circular_orbit(a: float, i: float, Omega: float, M: float):
    return Keplerian(a, 0, i, 0, Omega, M)

def propagate_orbit(orbit: Keplerian, time: float):
    mu = 398600.4418 # km^3/s^2
    a, e, i, omega, Omega, M = orbit
    n = np.sqrt(mu / a**3)
    M_new = M + n * time
    return Keplerian(a, e, i, omega, Omega, M_new)

# Task generation algorithm
# 0. Filter requests within radius
# 1: Increment macro time (1 minute?)
# 2: Generate new filtered requests within radius (radius min defined as min of horizon distance)
# 3. If they are signed oppositely, use binary searcb to find the closest point
# 4. Log time of request and angle, generate as new request

# First... probably should make a visualizer for it
t_now = datetime.datetime.now()
orbit = propagate_orbit(circular_orbit(a=R_E+400, i=np.deg2rad(45), Omega=0, M=0), (t_now - J2000).total_seconds())
task = np.array([0, 51.6]) # Latitude, Longitude

task_eci = ecef2eci(latlong2ecef(task), t_now)
orbit_eci = kepler2eci(orbit)

from matplotlib import animation
from matplotlib.widgets import Slider

# Create a figure and axis for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the plot with Earth as a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = R_E * np.outer(np.cos(u), np.sin(v))
y = R_E * np.outer(np.sin(u), np.sin(v))
z = R_E * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.3)

# Plot task on the surface of the Earth (this doesn't change over time)
task_eci = ecef2eci(latlong2ecef(task), t_now)
task_point = ax.scatter(task_eci[0], task_eci[1], task_eci[2], color='r', s=100, label="Task")

# Set labels and plot parameters
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Satellite Orbit and Task on Earth Surface')
ax.legend()

# Create a scatter object for the satellite, which will be updated during animation
satellite_point = ax.scatter([], [], [], color='g', s=100, label="Satellite")

# Define initial time delta in seconds
initial_seconds = 0

# Define the update function for the animation
def update_satellite(seconds_since_epoch):
    global satellite_point
    orbit = propagate_orbit(circular_orbit(a=R_E+400, i=np.deg2rad(45), Omega=0, M=0), seconds_since_epoch)
    orbit_eci = kepler2eci(orbit)
    satellite_point._offsets3d = (orbit_eci[0], orbit_eci[1], orbit_eci[2])
    return satellite_point,

# Function to be called when the slider is changed
def update(val):
    seconds_since_epoch = (t_now - J2000).total_seconds() + slider.val * 3600  # Slider controls hours
    update_satellite(seconds_since_epoch)
    fig.canvas.draw_idle()

# Set up slider
slider_ax = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(slider_ax, 'Time (Hours)', 0, 24, valinit=0, valstep=0.1)

# Attach the slider update function
slider.on_changed(update)

# Call the update function for animation
ani = animation.FuncAnimation(fig, update_satellite, frames=np.arange(0, 24 * 3600, 3600), interval=50, blit=False)

plt.show()