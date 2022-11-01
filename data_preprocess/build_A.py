import pickle
import torch
"""
    build adjacency matrix of road_graph
"""
data_path = '../data/'
road_graph = pickle.load(open(data_path + 'road_graph.pkl', 'rb'))
n = road_graph.number_of_nodes()
A = torch.eye(n)

adj = dict(road_graph.adj)
for k,v in adj.items():
    v = dict(v)
    for i in v.keys():
        A[k][i] = 1
        A[i][k] = 1


torch.save(A, data_path+'road_graph_pt/A.pt')