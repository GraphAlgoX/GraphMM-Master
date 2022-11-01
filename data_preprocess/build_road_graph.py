import re
import networkx as nx
import torch
import pickle
from utils import gps2grid, get_border, create_dir

MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG = get_border('../data/road.txt')
GRID_SIZE = 50

def read_road(path: str) -> dict:
    """
        read road.txt
    """
    link_nodes_dict = {}
    cr = re.compile(r"(\d*)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\|(.*)")
    with open(path, 'r') as f:
        for line in f.readlines():
            data = cr.findall(line)
            if len(data) != 0:
                link_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points, neighbors = data[0]
                if int(link_dir) == 0 or int(link_dir) == 1:
                    link_dir = 0
                if int(link_dir) == 2:
                    link_dir = 1
                neighbors = list(map(lambda x: x.split(','), neighbors.split(';')))
                points = list(map(lambda x: x.split(' '), points.split(',')))
                points = list(map(lambda x: [float(x[0]), float(x[1])], points))
                assert len(points) == int(vertex_count)
                link_nodes_dict[data[0][0]] = [link_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points,
                                               neighbors]
    return link_nodes_dict


def construct_road_graph(roads: dict):
    """
        build road_graph networkx
    """
    graph = nx.Graph()
    link_dict = {}
    for road in roads.values():
        link_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points, neighbors = road
        link_id = int(link_id)
        if int(link_dir) == 3:
            print(road)
        p1 = points[0]
        p2 = points[-1]
        grid_x1, grid_y1 = gps2grid(p1[1], p1[0], MIN_LAT=MIN_LAT, MIN_LNG=MIN_LNG)
        grid_x2, grid_y2 = gps2grid(p2[1], p2[0], MIN_LAT=MIN_LAT, MIN_LNG=MIN_LNG)

        link_dict[link_id] = [link_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points, neighbors]
        graph.add_node(link_id, x1=grid_x1, y1=grid_y1, x2=grid_x2, y2=grid_y2, lat1=p1[1], lng1=p1[0], lat2=p2[1],
                       lng2=p2[0], speed=speed, link_id=link_id)

    for road in roads.values():
        link_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points, neighbors = road
        for neighbor in neighbors:
            if neighbor == None or (len(neighbor) == 1 and neighbor[0] == ''):
                continue
            neighbor_link_id, __, _ = neighbor
            if int(neighbor_link_id) not in link_dict:
                continue
            my_id, neighbor_id = int(link_id), int(neighbor_link_id)   
            graph.add_edge(neighbor_id, my_id)
    return graph

def build_x_edge_index(G):
    """
        build feature and edge_index for road graph
    """
    x_feat = []
    num = G.number_of_nodes()
    
    for i in range(num):
        tmp_feat = []
        self_grid = []
        self_feat = dict(G.nodes[i])
        self_grid.append((self_feat['x1'], self_feat['y1']))
        self_grid.append((self_feat['x2'], self_feat['y2']))

        min_x, min_y = min(self_feat['x1'], self_feat['x2']), min(self_feat['y1'], self_feat['y2'])
        max_x, max_y = max(self_feat['x1'], self_feat['x2']), max(self_feat['y1'], self_feat['y2'])
        min_lat, min_lng = min(self_feat['lat1'], self_feat['lat2']), min(self_feat['lng1'], self_feat['lng2'])
        max_lat, max_lng = max(self_feat['lat1'], self_feat['lat2']), max(self_feat['lng1'], self_feat['lng2'])

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
        x_feat.append(tmp_feat)

    edge_index = [[], []]
    for i in G.edges():
        assert(int(i[0]) != int(i[1]))
        edge_index[0].append(int(i[0]))
        edge_index[1].append(int(i[1]))

        edge_index[0].append(int(i[1]))
        edge_index[1].append(int(i[0]))

    edge_index = torch.tensor(edge_index)
    x = torch.tensor(x_feat)
    return x, edge_index, G


if __name__ == '__main__':
    r = read_road('../data/road.txt')
    g = construct_road_graph(r)
    data_path = '../data/'
    x, edge_index, G = build_x_edge_index(g)
    pickle.dump(G, open(data_path+'road_graph.pkl', 'wb'))
    create_dir(data_path+'road_graph_pt')
    torch.save(x, data_path+'road_graph_pt/x.pt')
    torch.save(edge_index, data_path+'road_graph_pt/edge_index.pt')
