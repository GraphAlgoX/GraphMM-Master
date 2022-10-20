import torch
import json
map_matrix = torch.load('/data/GeQian/g2s_2/gmm_data/data/trace_graph_pt/' + 'map_matrix.pt')
data_path = '/data/GeQian/g2s_2/gmm_data/data/'
# val_data/val.json
name_ls = ['train', 'val', 'test']
for name in name_ls:
    path = data_path + name + '_data/' + name + '.json'
    x = json.load(open(path, 'r'))
    print(len(x))
print(map_matrix.shape)

