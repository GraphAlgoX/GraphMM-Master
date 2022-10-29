import networkx as nx
import json
import pickle
import torch
import os
import sys


# path = '/data/GeQian/g2s_2/map-matching-dataset/' + idx + '/'
# data_path = path + 'data/'

MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG = 360., 360., -360., -360.
GRID_SIZE = 50

def get_border(path):
    global MAX_LAT, MAX_LNG, MIN_LAT, MIN_LNG
    with open(path, 'r') as f:
        road_ls = f.readlines()
    for road in road_ls:
        road = road.strip()
        lng = float(road.split('\t')[0])
        lat = float(road.split('\t')[1])
        MAX_LAT = max(MAX_LAT, lat)
        MAX_LNG = max(MAX_LNG, lng)
        MIN_LAT = min(MIN_LAT, lat)
        MIN_LNG = min(MIN_LNG, lng)
    # print(MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG)

def gps2grid(lat, lng):
    lat_per_meter = 8.993203677616966e-06
    lng_per_meter = 1.1700193970443768e-05
    lat_unit = lat_per_meter * GRID_SIZE
    lng_unit = lng_per_meter * GRID_SIZE

    loc_grid_x = int((lat - MIN_LAT) / lat_unit) + 1
    loc_grid_y = int((lng - MIN_LNG) / lng_unit) + 1

    return loc_grid_x, loc_grid_y

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

def get_data(path):
    grid2id_dict = {}
    trace_dict = {}
    with open(path, 'r') as f:
        trace_ls = f.readlines()
 
    lst_id = -1
    for trace in trace_ls:
        trace = trace.strip()
        lng = float(trace.split('\t')[0])
        lat = float(trace.split('\t')[1])
        
        gridx, gridy = gps2grid(lat, lng)
        
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
    # print(len(trace_dict))
    return grid2id_dict, trace_dict


def build_graph(grid2id_dict, trace_dict):
    G = nx.DiGraph()
    weighted_edges = []
    for k, v in trace_dict.items():
        weighted_edges.append((k[0], k[1], v))
    G.add_weighted_edges_from(weighted_edges)
    # print('nodes num = ', G.number_of_nodes())
    for k, v in grid2id_dict.items():
        if v not in G.nodes():
            G.add_node(v)
        G.nodes[v]['gridx'] = k[0]
        G.nodes[v]['gridy'] = k[1]
    return G


def build_pyG(G):
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


    inweight = weight.copy()
    outweight = weight.copy()
    # print(edge_index)
    for i in G.nodes:
        x.append(grid2gps(G.nodes[i]['gridx'], G.nodes[i]['gridy'],G.nodes[i]['gridx'], G.nodes[i]['gridy']))

    out_edge_index = torch.tensor(edge_index)
    in_edge_index = out_edge_index[[1, 0]]
    x = torch.tensor(x)
    inweight = torch.tensor(inweight)
    outweight = torch.tensor(outweight)
    return x, in_edge_index, inweight, out_edge_index, outweight






if __name__ == "__main__":
    idx = str(sys.argv[1])
    idx = '000000' + idx
    if len(idx) < 8:
        idx = '0' + idx
    path = '/data/GeQian/g2s_2/map-matching-dataset/' + idx + '/'
    data_path = path + 'data/'
    get_border(path+idx+'.nodes')

    grid2id_dict, trace_dict = get_data(path+idx+'.track')
    # print(len(trace_dict), len(grid2id_dict))
    # print(max(list(grid2id_dict.values())))

    for k in trace_dict.keys():
        if k[0] == k[1]:
            print(k)
    G = build_graph(grid2id_dict, trace_dict)

    # print(G)
    # print('nodes num = ', G.number_of_nodes(), 'edges num = ', G.number_of_edges()) 

    nx.write_gml(G, data_path+"trace_graph.gml")
    x, in_edge_index, inweight, out_edge_index, outweight = build_pyG(G)
    if not os.path.exists(data_path+'trace_graph_pt/'):
        os.mkdir(data_path+'trace_graph_pt/')
    torch.save(in_edge_index, data_path+'trace_graph_pt/in_edge_index.pt')
    torch.save(x, data_path+'trace_graph_pt/x.pt')
    torch.save(inweight, data_path+'trace_graph_pt/inweight.pt')
    torch.save(outweight, data_path+'trace_graph_pt/outweight.pt')
    torch.save(out_edge_index, data_path+'trace_graph_pt/out_edge_index.pt')
