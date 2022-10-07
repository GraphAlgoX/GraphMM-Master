import networkx as nx
import torch
import pickle

data_path = '/data/GeQian/g2s_2/data_for_GMM-Master/data/'
pkl_path = '/data/GeQian/g2s_2/data_for_GMM-Master/used_pkl/'

road_graph = nx.read_gml(data_path + 'road_graph.gml', destringizer=int)
trace_graph = nx.read_gml(data_path + 'trace_graph.gml', destringizer=int)
grid_x = torch.load(data_path + 'trace_graph_pt/x.pt')
trace_graph_grid_set = set()
grid2traceid_dict = {}
traceid2road_dict = pickle.load(open(pkl_path + 'traceid2road_dict.pkl', 'rb'))

map_matrix = torch.zeros(trace_graph.number_of_nodes(),
                         road_graph.number_of_nodes())
for traceid, roads in traceid2road_dict.items():
    for road in roads:
        map_matrix[traceid, road] = 1
    map_matrix[traceid] /= len(roads)

singleton_grid_mask = []
singleton_grid_loction = []
for i in range(trace_graph.number_of_nodes()):
    if i not in traceid2road_dict.keys():
        singleton_grid_mask.append(i)
        singleton_grid_loction.append(grid_x[i])

singleton_grid_loction = torch.stack(singleton_grid_loction, dim=0)
singleton_grid_mask = torch.tensor(singleton_grid_mask, dtype=int)
torch.save(singleton_grid_loction,
           data_path + 'trace_graph_pt/singleton_grid_location.pt')
torch.save(singleton_grid_mask,
           data_path + 'trace_graph_pt/singleton_grid_mask.pt')
torch.save(map_matrix, data_path + 'trace_graph_pt/map_matrix.pt')
print(map_matrix)
