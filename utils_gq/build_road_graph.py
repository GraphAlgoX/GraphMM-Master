import re
import networkx as nx
import torch

min_lat, max_lat, min_lng, max_lng = float("inf"), float("-inf"), float("inf"), float("-inf")
GRID_SIZE = 50


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
                if int(link_dir) == 0 or int(link_dir) == 1:
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
        if link_id == '3625':
            print('stop!!!')
        if int(link_dir) == 3:
            print(road)
        p1 = points[0]
        p2 = points[-1]
        grid_x1, grid_y1 = gps2grid(p1[1], p1[0])
        grid_x2, grid_y2 = gps2grid(p2[1], p2[0])
        if (grid_x2-grid_x1)*(grid_y2-grid_y1) > 60:
            print('stop')
        link_id_2_new_id_dict[link_id] = now_id
        link_dict[link_id] = [now_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points, neighbors]
        graph.add_node(now_id, x1=grid_x1, y1=grid_y1, x2=grid_x2, y2=grid_y2, lat1=p1[1], lng1=p1[0], lat2=p2[1],
                       lng2=p2[0], speed=speed, link_id=link_id)
        now_id += 1

    for road in roads.values():
        link_id, s_node_id, e_node_id, link_dir, speed, vertex_count, points, neighbors = road
        for neighbor in neighbors:
            if neighbor == None or (len(neighbor) == 1 and neighbor[0] == ''):
                continue
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


def build_x_edge_index(path):
    x = []
    G = nx.read_gml(path)
    for i in G.nodes():
        x.append([G.nodes[i]['lat1'], G.nodes[i]['lng1'], G.nodes[i]['lat2'], G.nodes[i]['lng2']])

    edge_index = [[], []]
    for i in G.edges():
        edge_index[0].append(int(i[0]))
        edge_index[1].append(int(i[1]))
    print(G)

    edge_index = torch.tensor(edge_index)
    x = torch.tensor(x)
    return x, edge_index


if __name__ == '__main__':
    data_path = '/data/GeQian/g2s_2/data_for_GMM-Master/data/'
    r = read_road(data_path+'pure_data/newroad.txt')
    g = construct_road_graph(r)
    nx.write_gml(g, data_path+'road_graph.gml')
    x, edge_index = build_x_edge_index(path=data_path+'road_graph.gml')
    torch.save(x, data_path+'road_graph_pt/x.pt')
    torch.save(edge_index, data_path+'road_graph_pt/edge_index.pt')
