import numpy as np
import os
import re
from dynamic_tasker.access import Request
from dynamic_tasker.orbits import Keplerian

def load_worldcities(n=1000, random_subsample=False):
    accesses = [] 
    # Open the CSV and process the lines
    with open(os.path.join(data_dir, 'worldcities/worldcities.csv'), 'r', encoding='utf8') as f:
        f.readline()  # Skip the header line
        i = 0
        for line in f:
            parts = re.split(r'(?:",")|"', line)
            # Extract the latitude, longitude, and city name
            lat = float(parts[3])
            lon = float(parts[4])
            city = parts[2]
            # Create a new Request object and append it to the list
            accesses.append(Request(len(accesses), lat, lon, city))
            i += 1
            if i >= n:
                break

    return accesses

# def generate_equaterial_lookahead_cluster_scenario():
    # Generate both the orbits and accesses...
    # 1. Generate the orbits

# Generate random requests (size N)
def generate_requests(N):
    """
    Generate N Request objects with lat/long
    sampled uniformly over Earth's surface, using numpy.
    """
    # 1) sample z = sin(lat) uniformly in [-1,1]
    z = np.random.uniform(-1.0, 1.0, size=N)
    # 2) sample longitude angle θ uniformly in [0, 2π)
    theta = np.random.uniform(0.0, 2*np.pi, size=N)

    # convert to lat, lon in degrees
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(theta)
    # shift to [–180, +180)
    lon = np.where(lon > 180.0, lon - 360.0, lon)

    # build Request instances
    return [
        Request(i, float(lat[i]), float(lon[i]), f"request_{i}")
        for i in range(N)
    ]
