import networkx as nx
import torch
from torch_sparse import SparseTensor
import pickle
import os.path as osp


class GraphData():
    def __init__(self, root_path, data_path, layer, gamma, device) -> None:
        self.device = device
        # load trace graph and road graph
        if not root_path.endswith('/'):
            root_path += '/'
        if not data_path.endswith('/'):
            data_path += '/'
        road_graph = pickle.load(open(root_path + 'road_graph.pkl', 'rb'))
        trace_graph = nx.read_gml(data_path + 'trace_graph.gml',
                                  destringizer=int)
        self.num_roads = road_graph.number_of_nodes()
        self.num_grids = trace_graph.number_of_nodes()
        # load edge weight of trace graph and road graph
        trace_pt_path = data_path + 'trace_graph_pt/'
        road_pt_path = root_path + 'road_graph_pt/'
        # 2*num_of_edges
        self.trace_weight = torch.load(trace_pt_path + 'inweight.pt').float().to(device)
        self.trace_in_edge_index = torch.load(trace_pt_path + 'in_edge_index.pt').to(device)
        self.trace_out_edge_index = torch.load(trace_pt_path + 'out_edge_index.pt').to(device)
        road_edge_index = torch.load(road_pt_path + 'edge_index.pt')
        # construct sparse adj
        self.road_adj = SparseTensor(row=road_edge_index[0],
                                     col=road_edge_index[1],
                                     sparse_sizes=(self.num_roads,
                                                   self.num_roads)).to(device)
        self.road_x = torch.load(road_pt_path + 'x.pt').to(device)

        self.singleton_grid_mask = torch.load(
            trace_pt_path + 'singleton_grid_mask.pt').to(device)
        self.singleton_grid_location = torch.load(
            trace_pt_path + 'singleton_grid_location.pt').to(device)

        self.map_matrix = torch.load(trace_pt_path +
                                     'map_matrix.pt').to(device)
        # load map dictonary
        pkl_path = osp.join(data_path, 'used_pkl/')
        self.grid2traceid_dict = pickle.load(
            open(pkl_path + 'grid2traceid_dict.pkl', 'rb'))
        self.traceid2grid_dict = {
            v: k
            for k, v in self.grid2traceid_dict.items()
        }
        # gain A^k
        A = torch.load(road_pt_path+'A.pt')
        # A_list [n, n]
        self.A_list = self.get_adj_poly(A, layer, gamma)
    
    def get_adj_poly(self, A, layer, gamma):
        A_ = A.to(self.device)
        ans = A_.clone()
        for _ in range(layer-1):
            ans = ans @ A_
        ans[ans != 0] = 1.
        ans[ans == 0] = -gamma
        return ans


if __name__ == "__main__":
    pass
