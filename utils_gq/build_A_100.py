import pickle
import torch
import sys
idx = str(sys.argv[1])
idx = '000000' + idx
if len(idx) < 8:
    idx = '0' + idx
path = '/data/GeQian/g2s_2/map-matching-dataset/' + idx + '/'
data_path = path + 'data/'
# data_path = '/data/GeQian/g2s_2/data_for_GMM-Master/data/'
road_graph = pickle.load(open(data_path + 'road_graph.pkl', 'rb'))
n = road_graph.number_of_nodes()
A = torch.eye(n)

adj = dict(road_graph.adj)
for k,v in adj.items():
    v = dict(v)
    for i in v.keys():
        A[k][i] = 1
        A[i][k] = 1

print(A.shape)

torch.save(A, data_path+'A.pt')