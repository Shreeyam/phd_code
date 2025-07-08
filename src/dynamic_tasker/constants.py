# constants.py
import datetime

class Constants:
    # Time
    J2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
    seconds_per_day = 24 * 60 * 60

    
    # Earth
    R_E = 6378 # km
    mu = 398600.4418 # km^3/s^2
    ERA_J2000 = 280.46 # Earth rotation at J2000
    gamma=360.9856123035484 # Earth rotation rate, deg/day

    # Sun
