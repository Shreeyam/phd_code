import numpy as np
import datetime
import re
from pyscipopt import Model, quicksum
from dataclasses import dataclass
from typing import List, Callable

from dynamic_tasker.access import Request
from dynamic_tasker.orbits import Keplerian

@dataclass
class Spacecraft:
    orbit: Keplerian
    agility: Callable
    # TODO: Maybe add instrument types?

@dataclass
class Scenario:
    requests: List[Request]
    satellites: List[Spacecraft]

def load_worldcities(n=1000, random_subsample=False):
    accesses = [] 
    # Open the CSV and process the lines
    with open('../../data/worldcities/worldcities.csv', 'r', encoding='utf8') as f:
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

# def save_scenario(scenario, filename):
    # Use pickle
    

def greedy_schedule(accesses, requests, agility):
    schedule = []
    request_mask = np.zeros(len(requests))
    for i in accesses:
        if(len(schedule) == 0):
            schedule.append(i)
        else:
            if(request_mask[i.requestid] == 0 and i.time > datetime.timedelta(seconds=agility(i.angle - schedule[-1].angle)) + schedule[-1].time):
                schedule.append(i)
                request_mask[i.requestid] = 1

    return schedule

def milp_schedule(accesses, requests, agility):
    model = Model("Scheduler")
    x = {}
    for i, a in enumerate(accesses):
        x[i] = model.addVar(vtype="B", name=f"x_{a.requestid}_{a.time}")
    
    # Add constraints based on agility
    for i in range(len(accesses) - 1):
        for j in range(i + 1, len(accesses)):
            if(accesses[j].time < datetime.timedelta(seconds=agility(accesses[j].angle - accesses[i].angle)) + accesses[i].time):
                model.addCons(x[i] + x[j] <= 1)

    # Add objective
    model.setObjective(quicksum(x[i] * a.utility for i, a in enumerate(accesses)), "maximize")
    model.optimize()
    sol = model.getBestSol()
    schedule = []
    for i, a in enumerate(accesses):
        if(sol[x[i]] == 1):
            schedule.append(a)

    return schedule



def no_repair(tasks, requests, agility):
    return tasks

# TODO: get rid of the occluded mask
def greedy_schedulerepair(schedule, total_tasks, requests, occluded_mask, agility, allow_duplicates=False):
    schedule_copy = schedule.copy()
    request_mask = np.zeros(len(requests))
    for i in schedule:
        request_mask[i.requestid] = 1

    stop_iterating = False
    i = 0
    while(not stop_iterating):
        if(i == len(schedule_copy) - 1):
            break
        
        # Current node, looking ahead
        current = schedule_copy[i]
        next = schedule_copy[i + 1]
        next_next = schedule_copy[i + 2] if i + 2 < len(schedule_copy) else None

        total_index = total_tasks.index(next)
        if(occluded_mask[total_index] == 1):
            # Find the next non-occluded task
            next_nonoccluded = [x for x in total_tasks[i:] if occluded_mask[total_tasks.index(x)] == 0 and x.time > current.time]
            
            if(next_next):
                next_nonoccluded = [x for x in next_nonoccluded if x.time < next_next.time]

            if(len(next_nonoccluded) == 0):
                schedule_copy.pop(i)
                break

            for j in next_nonoccluded:
                if(j.time > datetime.timedelta(seconds=agility(j.angle - current.angle)) + current.time and (next_next) and 
                    next_next.time > datetime.timedelta(seconds=agility(next_next.angle - j.angle)) + j.time):
                    if(not allow_duplicates):
                        if(request_mask[j.requestid] == 1):
                            continue

                    schedule_copy.pop(i + 1)
                    schedule_copy.insert(i + 1, j)
                    request_mask[j.requestid] = 1

        i += 1

    return schedule_copy

def eval_scenario(scenario, initial_scheduler, repair_scheduler):
    pass