import os.path as osp
import pickle
import json
import torch
from matplotlib.pyplot import grid
GRID_SIZE= 50
def gps2grid(lat, lng, grid_size=GRID_SIZE):
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
    """
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    locgrid_x = int((lat - MIN_LAT) / lat_unit) + 1
    locgrid_y = int((lng - MIN_LNG) / lng_unit) + 1
    
    return locgrid_x, locgrid_y


root_path = '/data/GeQian/g2s_2/data_for_GMM-Master'
name = 'train'
data_path = osp.join(root_path, f"data/{name}_data/{name}.json")

map_path = osp.join(root_path, "used_pkl/grid2traceid_dict.pkl")

grid2traceid_dict = pickle.load(open(map_path, 'rb'))
traceid2grid_dict = {k:v for v,k in grid2traceid_dict.items()}
map_matrix = torch.load(root_path+'/data/trace_graph_pt/map_matrix.pt')
grid_num, road_num = map_matrix.shape
grid2road_dict = {}
for i in range(grid_num):
    road_id_ls = map_matrix[i].nonzero().squeeze(-1).numpy().tolist()
    for j in road_id_ls:
        gx, gy = traceid2grid_dict[i]
        if (gx, gy) not in grid2road_dict.keys():
            grid2road_dict[(gx, gy)] = []
        grid2road_dict[(gx, gy)].append(j)

all_min_gridx, all_max_gridx, all_min_gridy, all_max_gridy = 10000, -1, 10000, -1
for i in grid2traceid_dict.keys():
    gridx, gridy = i
    all_min_gridx = min(all_min_gridx, gridx)
    all_max_gridx = max(all_max_gridx, gridx)
    all_min_gridy = min(all_min_gridy, gridy)
    all_max_gridy = max(all_max_gridy, gridy)
print(all_min_gridx, all_max_gridx, all_min_gridy, all_max_gridy)

traces_ls = []
with open(data_path, "r") as fp:
    data = json.load(fp)

st_x, st_y = all_min_gridx, all_min_gridy
win_x, win_y = 8, 8

grid_ls = []
mid_grid_ls = []
# all_min_gridx, all_max_gridx, all_min_gridy, all_max_gridy = 10000, -1, 10000, -1
mid_idx_dict = {}
batch_size = 32
max_id_x, max_id_y = (all_max_gridx - st_x)//win_x, (all_max_gridy - st_y)//win_y
min_max_gridxy_trace_ls = []
for trace_idx, gps_ls in enumerate(data[0::3]):
    grids = []
    min_gridx, max_gridx, min_gridy, max_gridy = 10000, -1, 10000, -1
    for gps in gps_ls:
        gridx, gridy = gps2grid(gps[0], gps[1])
        grids.append((gridx, gridy))
        min_gridx = min(min_gridx, gridx)
        min_gridy = min(min_gridy, gridy)
        max_gridx = max(max_gridx, gridx)
        max_gridy = max(max_gridy, gridy)
    grid_ls.append(grids)
    mid_x, mid_y = (min_gridx+max_gridx)/2, (min_gridy+max_gridy)/2
    mid_grid_ls.append((mid_x, mid_y))
    id_x = (mid_x - st_x) // win_x
    id_y = (mid_y - st_y) // win_y
    # if id_x == 19:
    #    id_x = 18
    # if id_y == 19:
    #     id_y = 18
    if (id_x, id_y) not in mid_idx_dict.keys():
        mid_idx_dict[(id_x, id_y)] = []
    
    mid_idx_dict[(id_x, id_y)].append(trace_idx)
    min_max_gridxy_trace_ls.append((min_gridx, min_gridy, max_gridx, max_gridy))
    # all_min_gridx = min(all_min_gridx, min_gridx)
    # all_max_gridx = max(all_max_gridx, max_gridx)
    # all_min_gridy = min(all_min_gridy, min_gridy)
    # all_max_gridy = max(all_max_gridy, max_gridy)

# roads_ls = data[1::3]
# traces_gps_ls = data[0::3]
# sampleIdx_ls = data[2::3]
# 6, 153, 5, 148

full_batch_ls = []

need_num = batch_size
tmp_batch = []
for i in range(0, max_id_x + 1):
    for j in range(0, max_id_y + 1):
        if (i, j) not in mid_idx_dict.keys():
            continue
        tmp_trace_ls = mid_idx_dict[(i,j)]
        tmp_num = len(tmp_trace_ls)
        tmp_idx = 0
        while need_num <= tmp_num:
            tmp_batch += tmp_trace_ls[tmp_idx:tmp_idx+need_num]
            full_batch_ls.append(tmp_batch)
            tmp_batch = []
            tmp_idx += need_num
            tmp_num -= need_num
            need_num = batch_size
        tmp_batch += tmp_trace_ls[tmp_idx:]
        need_num -= tmp_num
full_batch_ls.append(tmp_batch)
print(len(full_batch_ls))




max_set_lens = 0
road_set_len_ls = []
for tmsfdidx, batch_ls in enumerate(full_batch_ls):
    assert(len(batch_ls) == batch_size or tmsfdidx==len(full_batch_ls)-1)
    road_set = set()
    for trace_idx in batch_ls:
        min_gridx, min_gridy, max_gridx, max_gridy = min_max_gridxy_trace_ls[trace_idx]
        for x in range(min_gridx, max_gridy + 1):
            for y in range(min_gridy, max_gridy + 1):
                if (x, y) in grid2road_dict.keys():
                    for q in grid2road_dict[(x,y)]:
                        road_set.add(q)
    
    max_set_lens = max(max_set_lens, len(road_set))
    # print(len(road_set))
    road_set_len_ls.append(len(road_set))

def check_len(ls, threshold):
    ans = 0
    for i in ls:
        if i > threshold:
            ans += 1
    print(f'threshold = {threshold}, num = {ans}')

print('batch_size=', batch_size, ';  max_set_lens=',max_set_lens)
check_len(road_set_len_ls, 2000)
check_len(road_set_len_ls, 1000)
check_len(road_set_len_ls, 500)
check_len(road_set_len_ls, 200)
check_len(road_set_len_ls, 100)
print(f'full_road_set num = {len(road_set_len_ls)}')

