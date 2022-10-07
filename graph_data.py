import networkx as nx
import torch
from torch_sparse import SparseTensor
import numpy as np
import pickle


class GraphData():
    def __init__(self, parent_path, layer, device) -> None:
        self.device = device
        data_path = parent_path + 'data/'
        # load trace graph and road graph
        road_graph = nx.read_gml(data_path + 'road_graph.gml',
                                 destringizer=int)
        trace_graph = nx.read_gml(data_path + 'trace_graph.gml',
                                  destringizer=int)
        self.num_roads = road_graph.number_of_nodes()
        self.num_grids = trace_graph.number_of_nodes()
        # load edge weight of trace graph and road graph
        trace_pt_path = data_path + 'trace_graph_pt/'
        road_pt_path = data_path + 'road_graph_pt/'
        # 2*num_of_edges
        trace_inweight = torch.load(trace_pt_path + 'inweight.pt')
        trace_outweight = torch.load(trace_pt_path + 'outweight.pt')
        trace_in_edge_index = torch.load(trace_pt_path + 'in_edge_index.pt')
        trace_out_edge_index = torch.load(trace_pt_path + 'out_edge_index.pt')
        road_edge_index = torch.load(road_pt_path + 'edge_index.pt')
        # construct sparse adj
        self.road_adj = SparseTensor(row=road_edge_index[0],
                                     col=road_edge_index[1],
                                     sparse_sizes=(self.num_roads,
                                                   self.num_roads)).to(device)
        self.trace_inadj = SparseTensor(
            row=trace_in_edge_index[0],
            col=trace_in_edge_index[1],
            value=trace_inweight,
            sparse_sizes=(self.num_grids, self.num_grids)).to(device)
        self.trace_outadj = SparseTensor(
            row=trace_out_edge_index[0],
            col=trace_out_edge_index[1],
            value=trace_outweight,
            sparse_sizes=(self.num_grids, self.num_grids)).to(device)
        # load initial features of road graph
        self.road_x = torch.load(road_pt_path + 'x.pt').to(device)
        self.singleton_grid_mask = torch.load(
            trace_pt_path + 'singleton_grid_mask.pt').to(device)
        self.singleton_grid_location = torch.load(
            trace_pt_path + 'singleton_grid_location.pt').to(device)
        self.map_matrix = torch.load(trace_pt_path +
                                     'map_matrix.pt').to(device)
        # load map dictonary
        pkl_path = parent_path + 'used_pkl/'
        self.grid2traceid_dict = pickle.load(
            open(pkl_path + 'grid2traceid_dict.pkl', 'rb'))
        self.traceid2grid_dict = {
            v: k
            for k, v in self.grid2traceid_dict.items()
        }
        # load f(A_R)
        A = torch.tensor(np.array(nx.adjacency_matrix(road_graph).todense()),
                         dtype=torch.float)
        self.A_list = self.get_adj_poly(A, layer) 
        # A_list [1, n, n]

    def get_adj_poly_old(self, A, layer):
        nR = self.num_roads
        A_list = []
        lstA = torch.eye(nR)
        A_list.append(lstA.clone())
        for _ in range(layer):
            lstA = torch.mm(lstA, A)
            A_list.append(lstA.clone())
        return torch.stack(A_list)
    
    def get_adj_poly(self, A, layer):
        A_ = (A + torch.eye(self.num_roads)).to(self.device)
        ans = A_.clone()
        for _ in range(layer-1):
            ans = ans@A_
        ans[ans!=0] = 1.
        return ans.unsqueeze(0)


if __name__ == "__main__":
    pass
