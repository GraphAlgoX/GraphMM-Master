import math
import re

import networkx as nx

min_lat, max_lat, min_lng, max_lng = float("inf"), float("-inf"), float("inf"), float("-inf")
GRID_SIZE = 50

DEGREES_TO_RADIANS = math.pi / 180
RADIANS_TO_DEGREES = 1 / DEGREES_TO_RADIANS
EARTH_MEAN_RADIUS_METER = 6371008.7714
DEG_TO_KM = DEGREES_TO_RADIANS * EARTH_MEAN_RADIUS_METER
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05


def gps2grid(lat, lng):
    lat_per_meter = 8.993203677616966e-06
    lng_per_meter = 1.1700193970443768e-05
    lat_unit = lat_per_meter * GRID_SIZE
    lng_unit = lng_per_meter * GRID_SIZE

    loc_grid_x = int((lat - min_lat) / lat_unit) + 1
    loc_grid_y = int((lng - min_lng) / lng_unit) + 1

    return loc_grid_x, loc_grid_y


def read_road(path: str) -> dict:
    link_nodes_dict = {}
    global min_lat, max_lat, min_lng, max_lng
    cr = re.compile(r"(\d*)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\|(.*)")
    with open(path, 'r') as f:
        for line in f.readlines():
            data = cr.findall(line)
            if len(data) != 0:
                link_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points, neighbors = data[0]
                if int(link_dir) == 0 or int(link_dir) ==  1:
                    link_dir = 0
                if int(link_dir) == 2:
                    link_dir = 1
                neighbors = list(map(lambda x: x.split(','), neighbors.split(';')))

                points = list(map(lambda x: x.split(' '), points.split(',')))
                points = list(map(lambda x: [float(x[0]), float(x[1])], points))
                for point in points:
                    if point[0] < min_lng:
                        min_lng = point[0]
                    if point[0] > max_lng:
                        max_lng = point[0]
                    if point[1] < min_lat:
                        min_lat = point[1]
                    if point[1] > max_lat:
                        max_lat = point[1]
                assert len(points) == int(vertex_count)
                link_nodes_dict[data[0][0]] = [link_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points,
                                               neighbors]
    return link_nodes_dict


def construct_road_graph(roads: dict):
    graph = nx.Graph()
    link_id_2_new_id_dict = {}
    link_dict = {}
    now_id = 0
    for road in roads.values():
        link_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points, neighbors = road
        if int(link_dir) == 3:
            print(road)
        p1 = points[0]
        p2 = points[-1]
        grid_x1, grid_y1 = gps2grid(p1[1], p1[0])
        grid_x2, grid_y2 = gps2grid(p2[1], p2[0])
        link_id_2_new_id_dict[link_id] = now_id
        link_dict[link_id] = [now_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points, neighbors]
        graph.add_node(now_id, x1=grid_x1, y1=grid_y1, x2=grid_x2, y2=grid_y2, lat1=p1[1], lng1=p1[0], lat2=p2[1],
                       lng2=p2[0], speed=speed, link_id=link_id)
        now_id += 1

    for road in roads.values():
        link_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points, neighbors = road
        for neighbor in neighbors:
            neighbor_link_id, associate_node_id, _ = neighbor
            if neighbor_link_id not in link_dict:
                continue
            neighbor_dir = link_dict[neighbor_link_id][3]
            
            my_id, neighbor_id = link_id_2_new_id_dict[link_id], link_id_2_new_id_dict[neighbor_link_id]
            neighbor_s_id, neighbor_e_id = link_dict[neighbor_link_id][1], link_dict[neighbor_link_id][2]
            operate_type = 0
            
            # graph.add_edge(my_id, neighbor_id)
            graph.add_edge(neighbor_id, my_id)
                
    return graph


def distance(a_lat, a_lng, b_lat, b_lng):
    """
    Calculate haversine distance between two GPS points in meters
    Args:
    -----
        a,b: SPoint class
    Returns:
    --------
        d: float. haversine distance in meter
    """
    if a_lat == b_lat and a_lng == b_lng:
        return 0.0
    delta_lat = math.radians(b_lat - a_lat)
    delta_lng = math.radians(b_lng - a_lng)
    h = math.sin(delta_lat / 2.0) * math.sin(delta_lat / 2.0) + math.cos(math.radians(a_lat)) * math.cos(
        math.radians(b_lat)) * math.sin(delta_lng / 2.0) * math.sin(delta_lng / 2.0)
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
    d = EARTH_MEAN_RADIUS_METER * c
    return d

def cal_degree(graph):
    max_in_degree = max(graph.degree, key=lambda x: x[1])[1]
    min_in_degree = min(graph.degree, key=lambda x: x[1])[1]
    max_out_degree = max(graph.degree, key=lambda x: x[1])[1]
    min_out_degree = min(graph.degree, key=lambda x: x[1])[1]
    avg_in_degree = sum([x[1] for x in graph.degree]) / graph.number_of_nodes()
    avg_out_degree = sum([x[1] for x in graph.degree]) / graph.number_of_nodes()
    print(f'max in degree: {max_in_degree}, min in degree: {min_in_degree}, avg in degree: {avg_in_degree}')
    print(f'max out degree: {max_out_degree}, min out degree: {min_out_degree}, avg out degree: {avg_out_degree}')


def cal_kth_avg_neighbor_nums(graph, k):
    total_num = 0
    for node in graph.nodes():
        subgraph = nx.ego_graph(graph, node, radius=k)
        total_num += subgraph.number_of_nodes()
    print(f'k={k}, avg neighbor nums: {total_num / graph.number_of_nodes()}')


# def check_len(threshold=160):

class SuperGraph:
    def __init__(self, g: nx.Graph, thre_degree=4, thre_lens=160):
        self.thre_degree = thre_degree
        self.thre_lens = thre_lens
        self.g = g

        self.feat = {}
        self.parent = {}
        self.neigh = {}
        self.degree = {}
        self.new_node_set = set()

        for k, v in dict(g.nodes).items():
            self.parent[k] = k
            self.feat[k] = {'l':distance(v['lat1'], v['lng1'], v['lat2'], v['lng2'])}

        for k, v in dict(g.degree).items():
            self.feat[k]['d'] = v
            self.neigh[k] = set()
            if v not in self.degree.keys():
                self.degree[v] = set()
            self.degree[v].add(k)
            
        for k,v in dict(self.g.adj).items():
            if k == v:
                print(f'exist self-loop {k}')
            for vi in v:
                self.neigh[k].add(vi)

        print('finished initing')   

    def remove_node(self, id):
        # assert(len(self.neigh[29871])==self.feat[29871]['d'])
        if id in self.feat.keys():
            if id in self.degree[self.feat[id]['d']]:
                self.degree[self.feat[id]['d']].remove(id)
            # del self.feat[id]
        for i in self.neigh[id]:
            if id in self.neigh[i]:
                self.neigh[i].remove(id)
                lstd = self.feat[i]['d']
                self.degree[lstd].remove(i)
                self.degree[lstd-1].add(i)
                self.feat[i]['d'] = lstd-1
        # assert(len(self.neigh[29871])==self.feat[29871]['d'])
    
    def check_len(self, k):
        v = self.feat[k]['l']
        if v >= self.thre_lens:
            self.new_node_set.add(k)
            self.remove_node(k)
    
    def merge(self, u, v): # u -> v
        # assert(len(self.neigh[29871])==self.feat[29871]['d'])
        for k, pk in self.parent.items():
            if pk == u:
                self.parent[k] = v

        self.degree[self.feat[u]['d']].remove(u)
        self.feat[v]['l'] += self.feat[u]['l']
        del self.feat[u]

        for i in self.neigh[u]:
            if i == v:
                self.neigh[v].remove(u)
                continue
            self.neigh[i].add(v)
            self.neigh[v].add(i)
            self.neigh[i].remove(u)
            self.degree[self.feat[i]['d']].remove(i)
            self.feat[i]['d'] = len(self.neigh[i])
            self.degree[self.feat[i]['d']].add(i)
            


        self.degree[self.feat[v]['d']].remove(v)
        self.feat[v]['d'] = len(self.neigh[v])
        self.degree[self.feat[v]['d']].add(v) 
        del self.neigh[u]

        # assert(len(self.neigh[29871])==self.feat[29871]['d'])

    def new_graph(self):
        nG = nx.Graph()
        child = {}
        for k,v in self.parent.items():
            if v not in child.keys():
                child[v] = []
            child[v].append(k)

        node_feat = dict(self.g.nodes)
        for k,v in child.items():
            nG.add_node(k, self_feat=node_feat[k])
            nG.nodes[k]['child_feat'] = []
            nG.nodes[k]['lens'] = self.feat[k]['l']
            for i in v:
                if i == k:
                    continue
                nG.nodes[k]['child_feat'].append(node_feat[i])
            
        for k,v in self.neigh.items():
            for i in v:
                nG.add_edge(self.parent[k], self.parent[i])
        return nG, child

        
        
        


    def solve(self):
        for k in list(self.feat.keys()):   
            self.check_len(k)
        while(True):
            d = 0
            restartflag = False
            for d in range(self.thre_degree):
                if d == 0:
                    continue
                # print('left num = ', len(self.neigh), ', d=', d)
                restartflag = False
                # assert(len(self.neigh[29871])==self.feat[29871]['d'])
                for id in self.degree[d]:
                    min_deg_idx = 0
                    min_deg = 9999
                    for nb in self.neigh[id]:   
                        if self.feat[nb]['d'] < min_deg:
                            min_deg = self.feat[nb]['d']
                            min_deg_idx = nb
                    self.merge(id, min_deg_idx)
                    self.check_len(min_deg_idx)
                    restartflag = True
                    break
                # assert(len(self.neigh[29871])==self.feat[29871]['d'])
                
                if restartflag:
                    break
            if d == self.thre_degree - 1 and not restartflag:
                break
        
        nG, child = self.new_graph()
        return nG, child, self.parent



if __name__ == '__main__':
    r = read_road('/data/GeQian/g2s_2/GNNMapMatch_haixu/data/road.txt')
    g = construct_road_graph(r)
    # cal_degree(g)
    # for i in (2, 3, 4):
    #     cal_kth_avg_neighbor_nums(g, i)

    lens_threshold = 160
    degree_threshold = 5
    sg = SuperGraph(g, degree_threshold, lens_threshold)
    nG, child, parent = sg.solve()
    print('new graph information: ')
    print(f'    lens_threshold = {lens_threshold}, degree_threshold = {degree_threshold}')
    print(f'    num of nodes = {nG.number_of_nodes()}')
    print(f'    num of edges = {nG.number_of_edges()}')
    
    print(f'    avg degree = ', nG.number_of_edges()/nG.number_of_nodes())
    full_lens = 0
    for k,v in dict(nG.nodes).items():
        full_lens += v['lens']
    
    avg_lens = full_lens/nG.number_of_nodes()
    print(f'    avg_lens = {avg_lens}')
