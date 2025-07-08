# access.py
import numpy as np
import datetime
import itertools
from tqdm import tqdm

from collections import namedtuple
from dynamic_tasker.orbits import *
from dynamic_tasker.vector import *
from dynamic_tasker.spacecraft import *

class Request:
    def __init__(self, id, lat, long, name, utility=1):
        self.id = id
        self.lat = lat
        self.long = long
        self.name = name
        self.utility = utility

    def __repr__(self):
        return f"Request({self.id}, {self.lat}, {self.long}, {self.name}, {self.utility})"

class Access:
    def __init__(self, request, time, angle, state=None, utility=None):
        self.request = request
        self.requestid = request.id
        self.lat = self.request.lat
        self.long = self.request.long
        self.name = self.request.name
        self.time = time
        self.angle = angle
        self.state = state

        if(utility is not None):
            self.utility = utility
        else:
            self.utility = self.request.utility

    def __repr__(self):
        return f"Access({self.requestid}, {self.lat}, {self.long}, {self.name}, {self.time}, {self.angle}, {self.state}, {self.utility})"

def multi_async_dispatch_search(requests, req_latlongs, field_of_regard, t0_s, t0, t1, t2, r1, r2, d1, d2, orbit):
    t3 = (t1 + t2) / 2
    # 1 second threshold
    if(t2 - t1 < 1):
        tasks = []
        for i, x in enumerate(requests):
            # Check access constraints...
            # Propagate to position
            task_eci = ecef2eci(latlong2ecef((x.lat, x.long)), t0 + datetime.timedelta(seconds=t3 - t0_s))
            r, v = kepler2eci(propagate_orbit(orbit, t1))
            # Compute the angle using dot product
            angle_diff = np.arccos(np.dot(r, r - task_eci) / (np.linalg.norm(r) * np.linalg.norm(task_eci - r)))
            sign = np.sign(dist2plane(r, np.cross(r, v), task_eci))   
            angle_diff = np.rad2deg(angle_diff) * sign
            if(np.abs(angle_diff) < field_of_regard and task_not_in_eclipse(t0 + datetime.timedelta(seconds=t3 - t0_s), r)):
                tasks.append(Access(x, t0 + datetime.timedelta(seconds=t3 - t0_s), angle_diff))
        
        return tasks
    
    # Assume all tasks change sign in this time period
    r3, v3 = kepler2eci(propagate_orbit(orbit, t3))

    tasks_eci_3 = ecef2eci_vec(latlong2ecef_vec(req_latlongs), t0 + datetime.timedelta(seconds=t3 - t0_s)) 

    d3 = dist2plane(r3, v3, tasks_eci_3)

    filter_firsthalf = d1 * d3 < 0
    filter_secondhalf = d2 * d3 < 0

    tasks_firsthalf = [requests[i] for i in range(len(requests)) if filter_firsthalf[i]]
    tasks_secondhalf = [requests[i] for i in range(len(requests)) if filter_secondhalf[i]]

    task_times_1 = []
    task_times_2 = []

    # Split and recurse
    if(len(tasks_firsthalf) > 0):
        task_times_1 = multi_async_dispatch_search(tasks_firsthalf, req_latlongs[filter_firsthalf], field_of_regard, t0_s, t0, t1, t3, r1, r3, d1[filter_firsthalf], d3[filter_firsthalf], orbit)

    if(len(tasks_secondhalf) > 0):
        task_times_2 = multi_async_dispatch_search(tasks_secondhalf, req_latlongs[filter_secondhalf], field_of_regard, t0_s, t0, t3, t2, r3, r2, d3[filter_secondhalf], d2[filter_secondhalf], orbit)
    
    return task_times_1 + task_times_2

def get_accesses(requests, orbit, t_coarse, field_of_regard, t0, t_end):
    h = orbit.a - Constants.R_E # assume circular orbit
    v = v_orb(h)
    theta = np.arctan((h * np.tan(np.deg2rad(field_of_regard))) / Constants.R_E) # Max possible lateral angle
    theta_total = theta + np.deg2rad((Constants.gamma / (24 * 3600)) * t_coarse) # Worst case Earth rotation
    filter_radius = np.sqrt((v * t_coarse/2)**2 + (h + Constants.R_E * (1 - np.cos(theta_total)))**2 + (Constants.R_E * np.sin(theta_total))**2)

    seconds_since_epoch = 0 #(t0 - Constants.J2000).total_seconds()

    total_tasks = []
    req_latlongs = np.array([[x.lat, x.long] for x in requests])

    r1, v1 = kepler2eci(propagate_orbit(orbit, seconds_since_epoch))
    for i in range(0, int((t_end - t0).total_seconds()) - t_coarse, t_coarse):
        t1 = seconds_since_epoch + i
        t2 = seconds_since_epoch + i + t_coarse
        
        # Calculate ECI points 
        r2, v2 = kepler2eci(propagate_orbit(orbit, t2))

        tasks_eci_1 = ecef2eci_vec(latlong2ecef_vec(req_latlongs), t0 + datetime.timedelta(seconds=i)) 
        tasks_eci_2 = ecef2eci_vec(latlong2ecef_vec(req_latlongs), t0 + datetime.timedelta(seconds=i + t_coarse)) 

        task_dists_1 = dist(r1, tasks_eci_1) 
        task_dists_2 = dist(r2, tasks_eci_2) 

        task_mask = (task_dists_1 <= filter_radius) + (task_dists_2 <= filter_radius)

        if(np.sum(task_mask) > 0):
            requests_prefiltered = list(itertools.compress(requests, task_mask))
            req_latlongs_prefiltered = list(itertools.compress(req_latlongs, task_mask))

            d1 = dist2plane(r1, v1, tasks_eci_1[task_mask])
            d2 = dist2plane(r2, v2, tasks_eci_2[task_mask])
            
            requests_crossindex = [d1[i] * d2[i] < 0 and d1[i] > d2[i] for i in range(len(requests_prefiltered))]

            if(np.sum(requests_crossindex) > 0):
                requests_filtered = list(itertools.compress(requests_prefiltered, requests_crossindex))
                req_latlongs_filtered = np.array(list(itertools.compress(req_latlongs_prefiltered, requests_crossindex)))
                new_tasks = multi_async_dispatch_search(requests_filtered, req_latlongs_filtered, field_of_regard, seconds_since_epoch, t0, t1, t2, r1, r2, d1[requests_crossindex], d2[requests_crossindex], orbit)
                total_tasks += new_tasks

        r1 = r2
        v1 = v2
    return total_tasks

def task_not_in_eclipse(time, r):
    sun_vector = sunvec_eci(time)
    return np.dot(r, sun_vector) > 0