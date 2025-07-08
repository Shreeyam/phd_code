# imagery.py
from collections import namedtuple
import numpy as np
import matplotlib as pyplot
import requests
import datetime
import os
import zipfile
import xml.etree.ElementTree as ET
from requests.auth import HTTPBasicAuth
import xarray as xr
# Linux only....
try:
    import pygrib
except ImportError:
    pass
import re
import pyproj
from pathlib import Path

dateformat = "%Y-%m-%d_%H%M%S"
data_dir = Path(__file__).resolve().parent.parent.parent / "data"

image_sources = {
    "goes_west": {
        "url": "https://noaa-goes18.s3.amazonaws.com/",
        "description": "GOES West (18)",
        "product": "ABI-L2-ACMF",
        "lon": -137.2,
    },
    "goes_east": {
        "url": "https://noaa-goes16.s3.amazonaws.com/",
        "description": "GOES East (16)",
        "product": "ABI-L2-ACMF",
        "lon": -75.2,
    },
    "meteosat_zds": {
        "url": "https://api.eumetsat.int/data/download/1.0.0",
        "description": "Meteosat Zero Degree Service (ZDS)",
        "product": "EO:EUM:DAT:MSG:CLM",
        "lon": 0,
    },
    "meteosat_iodc": {
        "url": "https://api.eumetsat.int/data/download/1.0.0",
        "description": "Meteosat Indian Ocean Data Coverage (IODC)",
        "product": "EO:EUM:DAT:MSG:CLM-IODC",
        "lon": 45.5,
    },
    "himawari": {
        "url": "https://noaa-himawari9.s3.amazonaws.com/",
        "description": "Himawari 9",
        "product": "AHI-L2-FLDK-Clouds",
        "lon": 140.7,
    },
}

def download_goes_image(time: datetime, url, product, savefile, causal=True):
    if(time.minute == 0):
        time = time - datetime.timedelta(hours=1)

    year = time.year
    hour = time.hour
    minute = time.minute


    # Calculate day of year
    day_of_year = time.timetuple().tm_yday

    response = requests.get(f"{url}/?prefix={product}/{year}/{day_of_year:03d}/{hour:02d}/", verify=False)
    response.raise_for_status()

    # Parse xml tree
    root = ET.fromstring(response.text)

    # Find the match
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    filenames = [content.find("s3:Key", namespace).text for content in root.findall("s3:Contents", namespace)] 
    # filename = [x for x in filenames if re.search(f"{minute:02d}[0-9]{{3}}\.nc", x)][0]
    filename = filenames[minute//10 if minute > 0 else -1]

    # Download the file
    response = requests.get(f"{url}/{filename}", verify=False)
    response.raise_for_status()  

    # Save the file
    with open(os.path.join(data_dir, f"products/{savefile}.nc"), "wb") as f:
        f.write(response.content)

def download_goes_east_image(time: datetime):
    download_goes_image(time, image_sources["goes_east"]["url"], image_sources["goes_east"]["product"], f"goes_east_{time.strftime(dateformat)}")

def download_goes_west_image(time: datetime):
    download_goes_image(time, image_sources["goes_west"]["url"], image_sources["goes_west"]["product"], f"goes_west_{time.strftime(dateformat)}")

def download_meteosat_zds_image(time: datetime):
    token = get_auth_token_meteosat()
    download_meteosat_image(time=time, url=image_sources["meteosat_zds"]["url"], auth_token=token)

def download_meteosat_iodc_image(time: datetime):
    token = get_auth_token_meteosat()
    download_meteosat_image(time=time, url=image_sources["meteosat_iodc"]["url"], product="iodc", auth_token=token)

def download_himawari_image(time: datetime, url=image_sources["himawari"]["url"], product=image_sources["himawari"]["product"], savefile=None):
    year = time.year
    month = time.month
    day = time.day
    hour = time.hour
    minute = time.minute

    if(savefile is None):
        savefile = f"himawari_{time.strftime(dateformat)}"

    response = requests.get(f"{url}/?prefix={product}/{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}", verify=False)
    response.raise_for_status()    

    # Parse xml tree
    root = ET.fromstring(response.text)

    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    filenames = [content.find("s3:Key", namespace).text for content in root.findall("s3:Contents", namespace)] 

    filename = [x for x in filenames if "AHI-CMSK" in x][0]

    # Download the file
    response = requests.get(f"{url}/{filename}", verify=False)
    response.raise_for_status()

    # Save the file
    with open(os.path.join(data_dir, f"products/{savefile}.nc"), "wb") as f:
        f.write(response.content)

MeteosatAuthToken = namedtuple("MeteosatAuthToken", ["token", "expires"])

def get_auth_token_meteosat():
    # Get the auth token for the meteosat endpoint
    url = "https://api.eumetsat.int/token"
    # TODO: don't commit the keys...
    # TODO: use environment variables
    key = "PEhTM7nI3LVgZ0V36RsRl6e5_eQa"
    secret = "vzvjhBJIgathAAnVuEGU9KzCXuka"

    data = {
        "grant_type": "client_credentials"
    }

    # Make the request
    response = requests.post(url, data=data, auth=HTTPBasicAuth(key, secret), verify=False)

    if(response.status_code == 200):
        json_response =  response.json()
        token = json_response["access_token"]
        expires = datetime.datetime.now() + datetime.timedelta(seconds=json_response["expires_in"])
        return MeteosatAuthToken(token, expires)
    else:
        raise Exception("Failed to get auth token for meteosat")
    
def revoke_auth_token_meteosat(auth_token: MeteosatAuthToken):
    url = "https://api.eumetsat.int/revoke"
    data = {
        "token": auth_token.token
    }

    response = requests.post(url, data=data, verify=False)

    if(response.status_code == 200):
        return True
    else:
        return False
    

def list_meteosat_images(auth_token: MeteosatAuthToken):
    # List the available images from the meteosat endpoint
    # url = 
    pass

def download_meteosat_image(url, auth_token: MeteosatAuthToken, time, product="zds", tmp_folder=os.path.join(data_dir, "tmp"), products_folder=os.path.join(data_dir, "products"), causal=True):
    collection = "EO:EUM:DAT:MSG:CLM"

    if(product=="iodc"):
        collection += "-IODC"

    # Replace with web-safe product name
    collection = collection.replace(":", "%3A")
    year = time.year
    month = time.month
    day = time.day
    hour = time.hour
    minute = time.minute

    # round minute to nearest 15
    # if(causal):
    #     minute = int(np.floor(minute/15.0)) * 15
    # else:
    #     minute = int(np.round(minute/15.0)) * 15

    url = image_sources["meteosat_zds"]["url"]

    url += f"/collections/{collection}/dates/{year}/{month:02d}/{day:02d}/times/{hour:02d}/{minute:02d}?access_token={auth_token.token}"
    response = requests.get(url, headers={"Authorization": f"Bearer {auth_token.token}"}, verify=False)
    if(response.status_code != 200):
        raise Exception(f"Failed to download image: {response.status_code}, {response.text}")

    product_filename = os.path.join(tmp_folder, f"meteosat_{product}_{time.strftime(dateformat)}.zip")

    with open(product_filename, "wb") as f:
        f.write(response.content)

    # Unzip and save to data folder...
    with zipfile.ZipFile(product_filename, "r") as z:
        z.extractall(tmp_folder)

    # Now save the product .grb to the data folder...
    file_start = f"MSG{2 if product=='iodc' else 3}-SEVI-MSGCLMK-0100-0100-{year}{month:02d}{day:02d}{hour:02d}{minute:02d}"
    # Search for the grb file
    for root, dirs, files in os.walk(tmp_folder):
        for file in files:
            if file.startswith(file_start) and file.endswith(".grb"):
                os.rename(os.path.join(root, file), os.path.join(products_folder, f"meteosat_{product}_{time.strftime(dateformat)}.grb"))
                break

image_sources = {
    "goes_west": {
        "url": "https://noaa-goes18.s3.amazonaws.com/",
        "description": "GOES West (18)",
        "product": "ABI-L2-ACMF",
        "lon": -137.2,
        "download": download_goes_west_image,
    },
    "goes_east": {
        "url": "https://noaa-goes16.s3.amazonaws.com/",
        "description": "GOES East (16)",
        "product": "ABI-L2-ACMF",
        "lon": -75.2,
        "download": download_goes_east_image,
    },
    "meteosat_zds": {
        "url": "https://api.eumetsat.int/data/download/1.0.0",
        "description": "Meteosat Zero Degree Service (ZDS)",
        "product": "EO:EUM:DAT:MSG:CLM",
        "lon": 0,
        "download": download_meteosat_zds_image,
    },
    "meteosat_iodc": {
        "url": "https://api.eumetsat.int/data/download/1.0.0",
        "description": "Meteosat Indian Ocean Data Coverage (IODC)",
        "product": "EO:EUM:DAT:MSG:CLM-IODC",
        "lon": 45.5,
        "download": download_meteosat_iodc_image,
    },
    "himawari": {
        "url": "https://noaa-himawari9.s3.amazonaws.com/",
        "description": "Himawari 9",
        "product": "AHI-L2-FLDK-Clouds",
        "lon": 140.7,
        "download": download_himawari_image,
    },
}

def get_closest_latlong_sample(data, lats, longs, point):
    lat, lon = point

    lats_1d = lats[:, lats.shape[0]//2]
    longs_1d = longs[longs.shape[1]//2, :]

    lat_idx = np.nanargmin(np.abs(lats_1d - lat))

    long_idx = np.nanargmin(np.abs(longs[lat_idx, :] - lon))

    # import matplotlib.pyplot as plt
    # plt.plot(np.abs(longs[lat_idx, :] - lon))
    # plt.show()

    return data[lat_idx, long_idx]

def extract_latlong_from_meteosat_grib(filename):
    grbs = pygrib.open(filename)
    cloud_mask = grbs[1].values
    lats = grbs[1].latitudes.reshape((3712, 3712))
    lons = grbs[1].longitudes.reshape((3712, 3712))

# def get_global_cloud_map(time: datetime):
#     # Get the global cloud map from the meteosat endpoint
#     auth_token = get_auth_token_meteosat()
#     download_meteosat_image(image_sources["meteosat_zds"]["url"], auth_token, time)

def load_bcm(filename, lazy_load=False):
    # Load the binary cloud mask file
    # Get pure filename
    pure_filename = os.path.basename(filename)
    full_filename = os.path.join(data_dir, "products", filename)
    if(lazy_load and not os.path.exists(full_filename)):
        # Extract the date from the filename
        date = datetime.datetime.strptime("_".join(pure_filename.split(".")[0].split("_")[-2:]), dateformat)
        image_source = "_".join(pure_filename.split("_")[:-2])
        
        # Download the image
        image_sources[image_source]["download"](date)
        # Fall through to loading step

    if(pure_filename.startswith("goes")):
        ds = xr.open_dataset(full_filename)
        data = ds['BCM']

        proj_info = ds['goes_imager_projection'].attrs
        h = proj_info['perspective_point_height']
        lon_origin = proj_info['longitude_of_projection_origin']
        sweep_angle_axis = proj_info['sweep_angle_axis']
        # Define the projection
        proj = pyproj.Proj(proj='geos', h=h, lon_0=lon_origin, sweep=sweep_angle_axis)

        # Get x and y coordinate arrays
        x = ds['x'].values * h  # Scaled by perspective_point_height
        y = ds['y'].values * h

        # Create a meshgrid of x, y
        x_mesh, y_mesh = np.meshgrid(x, y)

        # Compute lat/lon from x/y
        lon, lat = proj(x_mesh, y_mesh, inverse=True)

        return data.to_numpy(), lat, lon
    
    elif(pure_filename.startswith("meteosat")):
        grbs = pygrib.open(full_filename)
        data = grbs[1].values
        lat = grbs[1].latitudes.reshape((3712, 3712))
        lon = grbs[1].longitudes.reshape((3712, 3712))

        lat[data == 3] = np.nan
        lon[data == 3] = np.nan

        lon[lon > 180] -= 360

        data[data == 3] = np.nan
        data[~np.isnan(data)] = (data[~np.isnan(data)] == 2)

        return data[::-1, ::-1], lat, lon
    
    elif(pure_filename.startswith("himawari")):
        ds = xr.open_dataset(full_filename)
        data = ds['CloudMaskBinary'].values
        lat = ds["Latitude"].values
        lon = ds["Longitude"].values

        return data, lat, lon
    else:
        raise Exception("Unknown file type")
    
def sample_global_bcm(all_data, points):
    center_lons = [x["lon"] for x in image_sources.values()]
    center_lon_sorted = np.sort(center_lons)
    center_lon_cutoffs = (center_lon_sorted + np.roll(center_lon_sorted, 1)) / 2
    center_lon_cutoffs[0] -= 180

    # Create the points
    mask = np.zeros(len(points))
    for i, p in enumerate(points):
        lat, lon = p

        sat_idx = np.searchsorted(center_lon_cutoffs, lon)

        if sat_idx == 0:
            sat_idx = 5

        data, lats, lons = all_data[sat_idx - 1]

        mask[i] = get_closest_latlong_sample(data, lats, lons, p)
    
    return mask

def derive_global_bcm(time, all_data=None, N_lat=1000, N_lon=3000, max_lat=60):
    lat_grid = np.linspace(-max_lat, max_lat, N_lat)
    lon_grid = np.linspace(-180, 180, N_lon)

    lons, lats = np.meshgrid(lon_grid, lat_grid)

    points = np.stack([lats.flatten(), lons.flatten()], axis=1)

    if(all_data is None):
        all_data = [load_bcm(os.path.join(data_dir, f"products/{x}_{time.strftime(dateformat)}.{'grb' if x.startswith('meteosat') else 'nc'}"), lazy_load=True) for x in image_sources.keys()]

    # Sample all points...
    mask = sample_global_bcm(all_data, points).reshape(lats.shape)

    return mask, lats, lons

def load_global_bcm(time, N_lat=1000, N_lon=3000):
    # First check if the derived product exists
    filename = os.path.join(data_dir, f"derived/bcm_{time.strftime(dateformat)}.npz")
    if(os.path.exists(filename)):
        # Load the derived product
        ds = np.load(filename)
        data = ds['BCM']
        lats = ds['Latitude']
        lons = ds['Longitude']

        return data, lats, lons
    else:
        # Load the original products
        data, lats, lons = derive_global_bcm(time, None, N_lat, N_lon)
        # Save the derived product
        np.savez(filename, BCM=data, Latitude=lats, Longitude=lons)
        return data, lats, lons
    