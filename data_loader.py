import os.path as osp
import torch
import json
import pickle
from torch.utils.data import Dataset
import data_preprocess.utils as utils

class MyDataset(Dataset):
    def __init__(self, root_path, path, name):
        # parent_path = 'path'
        if not root_path.endswith('/'):
            root_path += '/'
        self.MIN_LAT, self.MIN_LNG, MAX_LAT, MAX_LNG = utils.get_border(root_path + 'road.txt')
        if not path.endswith('/'):
            path += '/'
        self.data_path = osp.join(path, f"{name}_data/{name}.json")
        self.map_path = osp.join(path, "used_pkl/grid2traceid_dict.pkl")
        self.buildingDataset(self.data_path)

    def buildingDataset(self, data_path):
        grid2traceid_dict = pickle.load(open(self.map_path, 'rb'))
        self.traces_ls = []
        with open(data_path, "r") as fp:
            data = json.load(fp)
            for gps_ls in data[0::3]:
                traces = []
                for gps in gps_ls:
                    gridx, gridy = utils.gps2grid(gps[0], gps[1], MIN_LAT=self.MIN_LAT, MIN_LNG=self.MIN_LNG)
                    traces.append(grid2traceid_dict[(gridx, gridy)] + 1)
                self.traces_ls.append(traces)
            self.roads_ls = data[1::3]
            self.traces_gps_ls = data[0::3]
            self.sampleIdx_ls = data[2::3]
        self.length = len(self.traces_ls)
        assert len(self.traces_ls) == len(self.roads_ls)

    def __getitem__(self, index):
        return self.traces_ls[index], self.roads_ls[index],\
            self.traces_gps_ls[index], self.sampleIdx_ls[index]

    def __len__(self):
        return self.length


def padding(batch):
    trace_lens = [len(sample[0]) for sample in batch]
    road_lens = [len(sample[1]) for sample in batch]
    max_tlen, max_rlen = max(trace_lens), max(road_lens)
    x, y, z, w = [], [], [], []
    # 0: [PAD]
    for sample in batch:
        x.append(sample[0] + [0] * (max_tlen - len(sample[0])))
        y.append(sample[1] + [-1] * (max_rlen - len(sample[1])))
        z.append(sample[2] + [[0, 0]] * (max_tlen - len(sample[2])))
        w.append(sample[3] + [-1] * (max_tlen - len(sample[3])))
    f = torch.LongTensor
    return f(x), f(y), torch.FloatTensor(z), f(w), trace_lens, road_lens


if __name__ == "__main__":
    pass
