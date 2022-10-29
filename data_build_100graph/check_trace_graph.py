from typing_extensions import dataclass_transform
import networkx as nx
import json
idx = '00000000'
path = '/data/GeQian/g2s_2/map-matching-dataset/' + idx + '/'
data_path = path + 'data/'
G = nx.read_gml(data_path+"trace_graph.gml")
print(G)