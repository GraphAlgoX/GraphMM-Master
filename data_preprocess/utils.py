import torch
import numpy as np
import math
import os

GRID_SIZE = 50
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05
DEGREES_TO_RADIANS = math.pi / 180
RADIANS_TO_DEGREES = 1 / DEGREES_TO_RADIANS
EARTH_MEAN_RADIUS_METER = 6371008.7714
DEG_TO_KM = DEGREES_TO_RADIANS * EARTH_MEAN_RADIUS_METER

def get_border(path):
    """
        get the min(max) LAT(LNG)
    """
    MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG = 360, 360, -360, -360
    with open(path, 'r') as f:
        road_ls = f.readlines()
    for road in road_ls:
        tmpa = road.split('\t')[6].split('|')[0]
        lng_lat_ls = tmpa.split(',')
        for lng_lat in lng_lat_ls:
            lng = float(lng_lat.split(' ')[0])
            lat = float(lng_lat.split(' ')[1])
            MAX_LAT = max(MAX_LAT, lat)
            MAX_LNG = max(MAX_LNG, lng)
            MIN_LAT = min(MIN_LAT, lat)
            MIN_LNG = min(MIN_LNG, lng)
    return MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG

def gps2grid(lat, lng, MIN_LAT, MIN_LNG, grid_size=GRID_SIZE):
    """
    mbr:
        MBR class.
    grid size:
        int. in meter
    """
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    locgrid_x = int((lat - MIN_LAT) / lat_unit) + 1
    locgrid_y = int((lng - MIN_LNG) / lng_unit) + 1
    
    return locgrid_x, locgrid_y

def gps2grid_batch(gps, grid_size=GRID_SIZE):
    MIN_LAT = 40.0200685
    MAX_LAT = 40.0982601
    MIN_LNG = 116.2628457
    MAX_LNG = 116.3528631
    GRID_SIZE = 50
    """
    mbr:
        MBR class.
    grid size:
        int. in meter
    gps B*2
    """
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    ans = torch.zeros(gps.shape)
    ans[:,0] = torch.floor((gps[:,0]-MIN_LAT)/lat_unit) + 1
    ans[:,1] = torch.floor((gps[:,1]-MIN_LNG)/lng_unit) + 1
    
    return ans


def grid2gps(gridx1, gridy1, gridx2, gridy2, MIN_LAT, MIN_LNG, grid_size=GRID_SIZE):
    """
        return gps for each grid
    """
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size
    lat1 = (gridx1 - 1) * lat_unit + MIN_LAT
    lng1 = (gridy1 - 1) * lng_unit + MIN_LNG

    lat2 = gridx2 * lat_unit + MIN_LAT
    lng2 = gridy2 * lng_unit + MIN_LNG

    return lat1, lng1, lat2, lng2

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)