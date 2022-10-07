import torch
import numpy as np
import random


MIN_LAT = 40.0200685
MAX_LAT = 40.0982601
MIN_LNG = 116.2628457
MAX_LNG = 116.3528631
GRID_SIZE = 50


def gps2grid(lat, lng, grid_size=GRID_SIZE):
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


def grid2gps(gridx1, gridy1, gridx2, gridy2, grid_size=GRID_SIZE):
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size
    lat1 = (gridx1 - 1) * lat_unit + MIN_LAT
    lng1 = (gridy1 - 1) * lng_unit + MIN_LNG

    lat2 = gridx2 * lat_unit + MIN_LAT
    lng2 = gridy2 * lng_unit + MIN_LNG

    return lat1, lng1, lat2, lng2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_2hop_un_neighbors(G, node_idx):
    if isinstance(node_idx, torch.Tensor):
        node_idx = node_idx.cpu().numpy().tolist()
    node_set = set()
    for i in node_idx:
        for j in G.adj[i]:
            node_set.add(j)
    node_ls = list(node_set)
    for i in node_ls:
        for j in G.adj[i]:
            node_set.add(j)
    for i in node_idx:
        node_set.add(i)

    neigh_ls = list(node_set)
    return neigh_ls


def get_2hop_in_neighbors(G, node_idx):
    if isinstance(node_idx, torch.Tensor):
        node_idx = node_idx.cpu().numpy().tolist()
    node_set = set()
    for i in node_idx:
        for j in G.adj[i]:
            node_set.add(j)
    node_ls = list(node_set)
    for i in node_ls:
        for j in G.adj[i]:
            node_set.add(j)
    for i in node_idx:
        node_set.add(i)

    neigh_ls = list(node_set)
    return neigh_ls
