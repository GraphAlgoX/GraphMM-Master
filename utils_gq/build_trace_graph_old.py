import networkx as nx
import torch
import os
import pickle

MIN_LAT = 90  # 纬度
MAX_LAT = 0
MIN_LNG = 180
MAX_LNG = 0
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


def get_data(path):
    grid2id_dict = {}
    trace_dict = {}
    with open(path, 'r') as f:
        trace_ls = f.readlines()
 
    lst_id = -1
    for trace in trace_ls:
        if trace.startswith('#'):
            lst_id = -1
            continue
        lng = float(trace.split(',')[2])
        lat = float(trace.split(',')[1])
        if lng == 0 or lat == 0:
            print(trace)
        if not (lat < MAX_LAT and lat > MIN_LAT and lng < MAX_LNG and lng > MIN_LNG):
            print(lat, lng)
        gridx, gridy = gps2grid(lat, lng)
        
        if gridx < 0 or gridy < 0:
            print(lat, lng, gridx, gridy)
        if (gridx, gridy) not in grid2id_dict.keys():
            grid2id_dict[(gridx, gridy)] = len(grid2id_dict)
        tmp_id = grid2id_dict[(gridx, gridy)]
        if lst_id != -1:
            if lst_id == tmp_id:
                continue
            if (lst_id, tmp_id) not in trace_dict.keys():
                trace_dict[(lst_id, tmp_id)] = 1
            else:
                trace_dict[(lst_id, tmp_id)] += 1
        lst_id = tmp_id
    # print(trace_dict)
    print(len(trace_dict))

    return grid2id_dict, trace_dict


def build_graph(grid2id_dict, trace_dict):
    G = nx.DiGraph()
    weighted_edges = []
    for k, v in trace_dict.items():
        weighted_edges.append((k[0], k[1], v))
    G.add_weighted_edges_from(weighted_edges)
    print('nodes num = ', G.number_of_nodes())
    for k, v in grid2id_dict.items():
        G.nodes[v]['gridx'] = k[0]
        G.nodes[v]['gridy'] = k[1]
    return G


def build_pyG(G):
    # x是节点特征矩阵，这里设为单位矩阵。
    # x = torch.eye(G.number_of_nodes(), dtype=torch.float)

    # adj是图G的邻接矩阵的稀疏表示，左边节点对代表一条边，右边是边的值，adj是对称矩阵。
    edge_index = [[], []]
    x = []
    weight = []
    # nodeinweight = [0] * G.number_of_nodes()
    # nodeoutweight = [0] * G.number_of_nodes()
    for i in G.edges():
        edge_index[0].append(i[0])
        edge_index[1].append(i[1])
        tmp_weight = G[i[0]][i[1]]['weight']
        weight.append(tmp_weight)
        # nodeinweight[i[1]] += tmp_weight
        # nodeoutweight[i[0]] += tmp_weight

    inweight = weight.copy()
    outweight = weight.copy()
    # print(edge_index)
    for i in G.nodes:
        x.append(grid2gps(G.nodes[i]['gridx'], G.nodes[i]['gridy'],G.nodes[i]['gridx'], G.nodes[i]['gridy']))

    # for i in range(len(weight)):
    #     x1 = edge_index[0][i]
    #     x2 = edge_index[1][i]
    #     inweight[i] = inweight[i]
    #     outweight[i] = outweight[i]
        # inweight[i] = inweight[i]/nodeinweight[x2]*G.in_degree(x2)/(G.in_degree(x2)+1)
        # outweight[i] = outweight[i]/nodeoutweight[x1]*G.out_degree(x1)/(G.out_degree(x1)+1)

    # for i in G.nodes:
    #     edge_index[0].append(i)
    #     edge_index[1].append(i)
    #     inweight.append(1/(G.in_degree(i)+1))
    #     outweight.append(1/(G.out_degree(i)+1))
        # inweight.append()
    out_edge_index = torch.tensor(edge_index)
    in_edge_index = out_edge_index[[1, 0]]
    x = torch.tensor(x)
    inweight = torch.tensor(inweight)
    outweight = torch.tensor(outweight)
    return x, in_edge_index, inweight, out_edge_index, outweight


data_path = '/data/GeQian/g2s_2/gmm_data/data/'
get_border(data_path+'pure_data/newroad.txt')
print('MIN_LAT=', MIN_LAT)
print('MAX_LAT=', MAX_LAT)
print('MIN_LNG=', MIN_LNG)
print('MAX_LNG=', MAX_LNG)
# exit(0)


if __name__ == "__main__":
    # 40.058076, 116.338461
    # 85,130
    full_grid2id_dict = pickle.load(open(''))
    grid2id_dict, trace_dict = get_data(data_path+'pure_data/full_trace_new.txt')
    print(len(trace_dict), len(grid2id_dict))
    print(max(list(grid2id_dict.values())))
    id_se = set()

    for k, v in trace_dict.items():
        id_se.add(k[0])
        id_se.add(k[1])
    print('id_se num = ', len(id_se))
    id_ls = sorted(list(id_se))
    id2newid = {}
    for idx, id in enumerate(id_ls):
        id2newid[id] = idx

    grid2id_dict_ = {}
    for k, v in grid2id_dict.items():
        if v in id2newid.keys():
            grid2id_dict_[k] = id2newid[v]
    trace_dict_ = {}
    for k, v in trace_dict.items():
        trace_dict_[(id2newid[k[0]], id2newid[k[1]])] = v

    trace_dict = trace_dict_
    grid2id_dict = grid2id_dict_
    for k in trace_dict.keys():
        if k[0] == k[1]:
            print(k)
    G = build_graph(grid2id_dict, trace_dict)

    # G=nx.read_gml('trace_graph.gml', destringizer=int)
    print(G)
    print('nodes num = ', G.number_of_nodes(), 'edges num = ', G.number_of_edges()) 
    # 17363 nodes, 61349 edges, directed, each edge is counted twice?
    # print(nx.number_connected_components(G))
    # print(G.is_connected())
    nx.write_gml(G, data_path+"trace_graph.gml")
    x, in_edge_index, inweight, out_edge_index, outweight = build_pyG(G)
    if not os.path.exists(data_path+'trace_graph_pt/'):
        os.mkdir(data_path+'trace_graph_pt/')
    torch.save(in_edge_index, data_path+'trace_graph_pt/in_edge_index.pt')
    torch.save(x, data_path+'trace_graph_pt/x.pt')
    torch.save(inweight, data_path+'trace_graph_pt/inweight.pt')
    torch.save(outweight, data_path+'trace_graph_pt/outweight.pt')
    torch.save(out_edge_index, data_path+'trace_graph_pt/out_edge_index.pt')
