
mu = 398600
h = 400
field_of_regard = 30 # degrees

def v_orb(h):
    return np.sqrt(mu / (h + 6378.1))

t_coarse = 60 # seconds
v = v_orb(h)
filter_radius = np.sqrt((v * t_coarse/2)**2 + (h ** 2) * (1 + np.sin(np.deg2rad(field_of_regard))**2))
print(f"filter radius: {filter_radius}")

# Task generation algorithm
# 0. Filter requests within radius
# 1: Increment macro time (1 minute?)
# 2: Generate new filtered requests within radius (radius min defined as min of horizon distance)
# 3. If they are signed oppositely, use binary searcb to find the closest point
# Use signed distance (x - r).dot(v)/||v|| to find the closest point
# 4. Log time of request and angle, generate as new request

orbit = circular_orbit(a=R_E+h, i=np.deg2rad(88), Omega=0, M=0)
# tasks = np.array([[40, 0]]) # Latitude, Longitude

request = namedtuple('Request', ['id', 'lat', 'long', 'name'])
task = namedtuple('Task', ['requestid', 'lat', 'long', 'name', 'time', 'angle'])

# Generate random requests...
np.random.seed(2)
requests = [request(i, np.random.uniform(-90, 90), np.random.uniform(-180, 180), f"Request {i}") for i in range(70)]
# requests = [request(0, 40, 0, 'test1'), request(1, 85, -5, 'test2')]
# requests = [request(i, i * 85/6, 0, f"Request {i}") for i in range(50)]

seconds_since_epoch = (t_0 - J2000).total_seconds()

def dist(x, y):
    return np.linalg.norm(x - y)

def dist2plane(r, n, p):
    return ((p - r).dot(n) / np.linalg.norm(n))

def multi_async_dispatch_search(requests, t1, t2, orbit):
    t3 = (t1 + t2) / 2
    # 1 second threshold
    if(t2 - t1 < 1):
        tasks = []
        for i, x in enumerate(requests):
            # Propagate to position
            task_eci = ecef2eci(lat2long2ecef((x.lat, x.long)), t_0 + datetime.timedelta(seconds=t3 - seconds_since_epoch))
            r, v = kepler2eci(propagate_orbit(orbit, t1))
            # Compute the angle using dot product
            angle_diff = np.arccos(np.dot(r, r - task_eci) / (np.linalg.norm(r) * np.linalg.norm(task_eci - r)))
            sign = np.sign(dist2plane(r, np.cross(r, v), task_eci))   
            angle_diff = np.rad2deg(angle_diff) * sign
            tasks.append((t3, angle_diff))
        
        return tasks
    
    # Assume all tasks change sign in this time period
    r1, v1 = kepler2eci(propagate_orbit(orbit, t1))
    r2, v2 = kepler2eci(propagate_orbit(orbit, t2))
    r3, v3 = kepler2eci(propagate_orbit(orbit, t3))

    tasks_eci_1 = [ecef2eci(lat2long2ecef((x.lat, x.long)), t_0 + datetime.timedelta(seconds=t1 - seconds_since_epoch)) for x in requests]
    tasks_eci_2 = [ecef2eci(lat2long2ecef((x.lat, x.long)), t_0 + datetime.timedelta(seconds=t2 - seconds_since_epoch)) for x in requests]
    tasks_eci_3 = [ecef2eci(lat2long2ecef((x.lat, x.long)), t_0 + datetime.timedelta(seconds=t3 - seconds_since_epoch)) for x in requests]

    d1 = dist2plane(r1, v1, tasks_eci_1)
    d2 = dist2plane(r2, v2, tasks_eci_2)
    d3 = dist2plane(r3, v3, tasks_eci_3)

    filter_firsthalf = d1 * d3 < 0
    filter_secondhalf = d2 * d3 < 0

    tasks_firsthalf = [requests[i] for i in range(len(requests)) if filter_firsthalf[i]]
    tasks_secondhalf = [requests[i] for i in range(len(requests)) if filter_secondhalf[i]]

    task_times_1 = []
    task_times_2 = []

    # Split and recurse
    if(len(tasks_firsthalf) > 0):
        task_times_1 = multi_async_dispatch_search(tasks_firsthalf, t1, t3, orbit)

    if(len(tasks_secondhalf) > 0):
        task_times_2 = multi_async_dispatch_search(tasks_secondhalf, t3, t2, orbit)
    
    return task_times_1 + task_times_2

def get_total_tasks():
    total_tasks = []

    for i in tqdm(range(0, 7 * 24 * 3600 - t_coarse, t_coarse)):
        t1 = seconds_since_epoch + i
        t2 = seconds_since_epoch + i + t_coarse
        
        # Calculate ECI points 
        r1, v1 = kepler2eci(propagate_orbit(orbit, t1))
        r2, v2 = kepler2eci(propagate_orbit(orbit, t2))

        tasks_eci_1 = np.array([ecef2eci(lat2long2ecef((x.lat, x.long)), t_0 + datetime.timedelta(seconds=i)) for x in requests])
        tasks_eci_2 = np.array([ecef2eci(lat2long2ecef((x.lat, x.long)), t_0 + datetime.timedelta(seconds=i + t_coarse)) for x in requests])

        task_dists_1 = np.array([dist(r1, x) for x in tasks_eci_1])
        task_dists_2 = np.array([dist(r2, x) for x in tasks_eci_2])

        task_mask = (task_dists_1 <= filter_radius) + (task_dists_2 <= filter_radius)

        if(np.sum(task_mask) > 0):
            # Calculate distance to plane
            d1 = dist2plane(r1, v1, tasks_eci_1[task_mask])
            d2 = dist2plane(r2, v2, tasks_eci_2[task_mask])
            # Filter for sign changes
            requests_prefiltered = [requests[i] for i in range(len(requests)) if task_mask[i]]
            # If they are signed oppositely, use binary search to find the closest point
            requests_crossindex = [i for i in range(len(requests_prefiltered)) if d1[i] * d2[i] < 0]
            if(len(requests_crossindex) > 0):
                # print("Binary search")
                requests_filtered = [requests_prefiltered[i] for i in requests_crossindex]
                task_times = multi_async_dispatch_search(requests_filtered, t1, t2, orbit)
                new_tasks = [
                    task(
                        requests_filtered[i].id, 
                        requests_filtered[i].lat, 
                        requests_filtered[i].long, 
                        requests_filtered[i].name,
                        t_0 + datetime.timedelta(seconds=task_times[i][0] - seconds_since_epoch),
                        angle=task_times[i][1]
                    ) 
                    for i in range(len(requests_filtered))
                ]            
                # print(new_tasks)
                [total_tasks.append(x) for x in new_tasks]

    return total_tasks

total_tasks = get_total_tasks()
# %lprun -f get_total_tasks get_total_tasks()
