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



