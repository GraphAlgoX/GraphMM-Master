
# remove trace only in one grid

MIN_LAT, MAX_LAT, MIN_LNG, MAX_LNG = 40.0200685, 40.0982601, 116.2628457, 116.3528631
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
    lat1 = (gridx1-1)*lat_unit + MIN_LAT
    lng1 = (gridy1-1)*lng_unit + MIN_LNG

    lat2 = gridx2*lat_unit + MIN_LAT
    lng2 = gridy2*lng_unit + MIN_LNG

    return lat1, lng1, lat2, lng2


def get_border(path):
    global MAX_LAT, MAX_LNG, MIN_LAT, MIN_LNG
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


def readTrajFile(filePath):
    with open(filePath, 'r') as f:
        traj_list = f.readlines()
    finalLs = list()  # 用来保存所有轨迹
    tempLs = list()  # 用来保存单个轨迹
    for idx, sen in enumerate(traj_list):
        if sen[0] == '#':  # 表明是一条轨迹的开头
            if(idx != 0):
                finalLs.append(tempLs)
            tempLs = [sen]
        else:  # 增加轨迹点
            tempLs.append(sen)
    return finalLs


def clean(tmpLs, file):
    finalLs = []
    print(len(tmpLs))
    for traces in tmpLs:
        if len(traces) < 16:
            continue
        tmp_traces = traces.copy()
        grid_set = set()
        add = True
        for trace in traces:
            if trace[0] == '#':
                continue
            lng = float(trace.split(',')[2])
            lat = float(trace.split(',')[1])
            if lng == 0 or lat == 0:
                print(trace)
                add = False
                break
            if not (lat < MAX_LAT and lat > MIN_LAT and lng < MAX_LNG and lng > MIN_LNG):
                add = False
                print(lat, lng)
                break
            if trace[:4] != '2022':
                add = False
                print(trace)
                break
            gridx, gridy = gps2grid(lat, lng)
            grid_set.add((gridx, gridy))
        if len(grid_set) > 1 and add:
            finalLs.append(tmp_traces)
    print(len(finalLs))
    with open(file, 'w') as f:
        for traces in finalLs:
            for trace in traces:
                f.write(trace)


data_path = '/data/GeQian/g2s_2/gmm_data/data/'
get_border(data_path+'pure_data/newroad.txt')
# print('MIN_LAT=', MIN_LAT)
# print('MAX_LAT=', MAX_LAT)
# print('MIN_LNG=', MIN_LNG)
# print('MAX_LNG=', MAX_LNG)


if __name__ == "__main__":
    # 40.058076, 116.338461
    # 85,130
    tmpLs = readTrajFile('/data/GeQian/g2s_2/gmm_data/data/pure_data/full_trace_new.txt')
    clean(tmpLs, '/data/GeQian/g2s_2/gmm_data/data/pure_data/full_trace_new.txt')