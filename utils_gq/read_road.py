import networkx as nx
import pickle
import json
import torch
link2idx = json.load(open('/data/GeQian/g2s_2/data_for_GMM-Master/data/peeling_data/link2idx.json', 'r'))
data_path = '/data/GeQian/g2s_2/data_for_GMM-Master/data/'
# agraph = nx.read_gml(data_path+'road_graph.gml', destringizer=int)
road_graph = pickle.load(open('/data/GeQian/g2s_2/data_for_GMM-Master/data/peeling_data/newnG.pkl', 'rb'))
# graph = pickle.load(open('/data/GeQian/g2s_2/data_for_GMM-Master/data/road_graph.gml'))
old_x = torch.load('/data/GeQian/g2s_2/data_for_GMM-Master_old_1010/data/road_graph_pt/x.pt')
print(old_x)
road_graph = pickle.load(open('/data/GeQian/g2s_2/data_for_GMM-Master/data/road_graph.pkl', 'rb'))
# 8123 467
x = torch.load('/data/GeQian/g2s_2/data_for_GMM-Master/data/road_graph_pt/x.pt')
trace_graph = nx.read_gml(data_path+'trace_graph.gml', destringizer=int)
print(x)
# print(road_graph)