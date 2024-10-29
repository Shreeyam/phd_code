import numpy as np
import datetime

def greedy_schedule(tasks, requests, agility):
    schedule = []
    request_mask = np.zeros(len(requests))
    for i in tasks:
        if(len(schedule) == 0):
            schedule.append(i)
        else:
            if(request_mask[i.requestid] == 0 and i.time > datetime.timedelta(seconds=agility(i.angle - schedule[-1].angle)) + schedule[-1].time):
                schedule.append(i)
                request_mask[i.requestid] = 1

    return schedule

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