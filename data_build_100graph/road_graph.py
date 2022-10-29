import networkx as nx
import json
import pickle
import torch
import os
import sys

min_lat, min_lng, max_lat, max_lng = 360, 360, -360, -360
GRID_SIZE = 50

def gps2grid(lat, lng):
    lat_per_meter = 8.993203677616966e-06
    lng_per_meter = 1.1700193970443768e-05
    lat_unit = lat_per_meter * GRID_SIZE
    lng_unit = lng_per_meter * GRID_SIZE

    loc_grid_x = int((lat - min_lat) / lat_unit) + 1
    loc_grid_y = int((lng - min_lng) / lng_unit) + 1

    return loc_grid_x, loc_grid_y

# print(road_neighs)

def get_min_max(nodes):
    global min_lat, min_lng, max_lat, max_lng
    for k, v in nodes.items():
        min_lat = min(min_lat, v['lat'])
        max_lat = max(max_lat, v['lat'])

        min_lng = min(min_lng, v['lng'])
        max_lng = max(max_lng, v['lng'])

    print(min_lat, min_lng, max_lat, max_lng)

def construct_road_graph(road_neighs: dict, roads: dict, nodes: dict):
    graph = nx.Graph()
    for k, v in roads.items():
        snode, tnode = v[0], v[1]
        gridx1, gridy1 = gps2grid(lat=nodes[snode]['lat'], lng=nodes[snode]['lng'])
        gridx2, gridy2 = gps2grid(lat=nodes[tnode]['lat'], lng=nodes[tnode]['lng'])
        graph.add_node(k, lat1=nodes[snode]['lat'], lat2=nodes[tnode]['lat'], \
            lng1=nodes[snode]['lng'], lng2=nodes[tnode]['lng'], x1=gridx1, y1=gridy1, \
            x2=gridx2, y2=gridy2)
    
    for k, v in road_neighs.items():
        for i in v:
            graph.add_edge(k, i)
    return graph

def build_x_edge_index(G):
    x_feat = []
    # G = nx.read_gml(path)
    num = G.number_of_nodes()
    
    for i in range(num):
        tmp_feat = []
        self_grid = []
        self_feat = dict(G.nodes[i])
        self_grid.append((self_feat['x1'], self_feat['y1']))
        self_grid.append((self_feat['x2'], self_feat['y2']))
        self_lat_lng = []
        self_lat_lng += [(self_feat['lat1'], self_feat['lng1']), (self_feat['lat2'], self_feat['lng2'])]

        min_x, min_y = self_grid[0]
        max_x, max_y = self_grid[0]

        min_lat, min_lng = self_lat_lng[0]
        max_lat, max_lng = self_lat_lng[0]

        for j in self_grid[1:]:
            min_x = min(min_x, j[0])
            max_x = max(max_x, j[0])

            min_y = min(min_y, j[1])
            max_y = max(max_y, j[1])
        
        for j in self_lat_lng[1:]:
            min_lat = min(min_lat, j[0])
            max_lat = max(max_lat, j[0])

            min_lng = min(min_lng, j[1])
            max_lng = max(max_lng, j[1])

        

        xy_ls = []
        xy_ls += [(min_x, min_y), (max_x, max_y), ((min_x+max_x)//2, min_y), ((min_x+max_x)//2, max_y)]
        xy_ls += [(max_x, min_y), (min_x, max_y), (min_x, (min_y+max_y)//2), (max_x, (min_y+max_y)//2)]

        for xy in xy_ls:
            x,y = xy
            min_dis = 1000
            for self_xy in self_grid:
                sx, sy = self_xy
                min_dis = min(min_dis, abs(sx-x)+abs(sy-y))

            tmp_feat += [x,y,min_dis]
        
        tmp_feat += [min_lat, min_lng, max_lat, max_lng]
        G.nodes[i]['x1'], G.nodes[i]['x2'], G.nodes[i]['y1'], G.nodes[i]['y2'] = min_x, max_x, min_y, max_y
        # for xy in xy_ls:
        #     for i in 
        x_feat.append(tmp_feat)

    edge_index = [[], []]
    for i in G.edges():
        assert(int(i[0]) != int(i[1]))
        edge_index[0].append(int(i[0]))
        edge_index[1].append(int(i[1]))

        edge_index[0].append(int(i[1]))
        edge_index[1].append(int(i[0]))

    # print(G)

    edge_index = torch.tensor(edge_index)
    x = torch.tensor(x_feat)
    return x, edge_index, G


if __name__ == '__main__':
    idx = str(sys.argv[1])
    idx = '000000' + idx
    if len(idx) < 8:
        idx = '0' + idx
    path = '/data/GeQian/g2s_2/map-matching-dataset/' + idx + '/'
    nodes = {}

    with open(path + idx + '.nodes', 'r') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            lat, lng = float(line.split('\t')[1]), float(line.split('\t')[0])
            nodes[i] = {'lng':lng, 'lat':lat, 'road_ls':[]}

    roads = {}
    with open(path + idx + '.arcs', 'r') as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            node_ls = line.split('\t')
            snodeid = int(node_ls[0])
            tnodeid = int(node_ls[1])
            roads[i] = [snodeid, tnodeid]
            nodes[snodeid]['road_ls'].append(i)
            nodes[tnodeid]['road_ls'].append(i)

    road_neighs = {}
    for id, node in nodes.items():
        road_ls = node['road_ls']
        for j in road_ls:
            if j not in road_neighs.keys():
                road_neighs[j] = set()
            for k in road_ls:
                if j != k:
                    road_neighs[j].add(k)
    get_min_max(nodes)
    road_graph = construct_road_graph(road_neighs, roads, nodes)

    # print(road_graph)
    data_path = path + 'data/'

    x, edge_index, G = build_x_edge_index(road_graph)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(data_path+'road_graph_pt/'):
        os.mkdir(data_path+'road_graph_pt/')
    pickle.dump(G, open(data_path+'road_graph.pkl', 'wb'))
    torch.save(x, data_path+'road_graph_pt/x.pt')
    torch.save(edge_index, data_path+'road_graph_pt/edge_index.pt')
    print('road_graph finish')