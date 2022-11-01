import networkx as nx
import torch
import os
import sys
from utils import gps2grid, grid2gps, create_dir, get_border

MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG = get_border('../data/road.txt')
GRID_SIZE = 50

def get_data(path):
    """
        read trace.txt
    """
    grid2id_dict = {}
    trace_dict = {}
    with open(path, 'r') as f:
        trace_ls = f.readlines()
 
    lst_id = -1
    for trace in trace_ls:
        if trace.startswith('#'):
            lst_id = -1
            continue
        lng = float(trace.split(',')[2])
        lat = float(trace.split(',')[1])

        gridx, gridy = gps2grid(lat, lng, MIN_LAT=MIN_LAT, MIN_LNG=MIN_LNG)        
        if gridx < 0 or gridy < 0:
            print(lat, lng, gridx, gridy)
        if (gridx, gridy) not in grid2id_dict.keys():
            grid2id_dict[(gridx, gridy)] = len(grid2id_dict)
        tmp_id = grid2id_dict[(gridx, gridy)]
        if lst_id != -1:
            if lst_id == tmp_id:
                continue
            if (lst_id, tmp_id) not in trace_dict.keys():
                trace_dict[(lst_id, tmp_id)] = 1
            else:
                trace_dict[(lst_id, tmp_id)] += 1
        lst_id = tmp_id
    return grid2id_dict, trace_dict


def build_graph(grid2id_dict, trace_dict):
    """
        build trace graph
    """
    G = nx.DiGraph()
    weighted_edges = []
    for k, v in trace_dict.items():
        weighted_edges.append((k[0], k[1], v))
    G.add_weighted_edges_from(weighted_edges)
    for k, v in grid2id_dict.items():
        if v not in G.nodes():
            G.add_node(v)
        G.nodes[v]['gridx'] = k[0]
        G.nodes[v]['gridy'] = k[1]
    return G


def build_pyG(G):
    """
        build feature and edge_index for trace graph
    """
    edge_index = [[], []]
    x = []
    weight = []
    for i in G.edges():
        edge_index[0].append(i[0])
        edge_index[1].append(i[1])
        tmp_weight = G[i[0]][i[1]]['weight']
        weight.append(tmp_weight)

    inweight = weight.copy()
    outweight = weight.copy()

    for i in G.nodes:
        x.append(grid2gps(G.nodes[i]['gridx'], G.nodes[i]['gridy'],G.nodes[i]['gridx'], G.nodes[i]['gridy'], \
            MIN_LAT=MIN_LAT, MIN_LNG=MIN_LNG))

    out_edge_index = torch.tensor(edge_index)
    in_edge_index = out_edge_index[[1, 0]]
    x = torch.tensor(x)
    inweight = torch.tensor(inweight)
    outweight = torch.tensor(outweight)
    return x, in_edge_index, inweight, out_edge_index, outweight


if __name__ == "__main__":
    downsample_rate = sys.argv[1]
    path = '../data/'
    data_path = path + 'data' + downsample_rate + '/'
    grid2id_dict, trace_dict = get_data(data_path+'data_split/downsample_trace.txt')
    G = build_graph(grid2id_dict, trace_dict)
    nx.write_gml(G, data_path+"trace_graph.gml")
    x, in_edge_index, inweight, out_edge_index, outweight = build_pyG(G)
    create_dir(data_path+'trace_graph_pt/')
    torch.save(in_edge_index, data_path+'trace_graph_pt/in_edge_index.pt')
    torch.save(x, data_path+'trace_graph_pt/x.pt')
    torch.save(inweight, data_path+'trace_graph_pt/inweight.pt')
    torch.save(outweight, data_path+'trace_graph_pt/outweight.pt')
    torch.save(out_edge_index, data_path+'trace_graph_pt/out_edge_index.pt')
