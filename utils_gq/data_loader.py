import os.path as osp
import torch
import json
import pickle
from torch.utils.data import Dataset
import utils.utils as utils


class MyDataset(Dataset):
    def __init__(self, root_path, name):
        self.data_path = osp.join(root_path, f"data/{name}_data/{name}.json")
        self.map_path = osp.join(root_path, "used_pkl/grid2traceid_dict.pkl")
        self.buildingDataset(self.data_path)

    def buildingDataset(self, data_path):
        grid2traceid_dict = pickle.load(open(self.map_path, 'rb'))
        self.traces_ls = []
        # self.traces_gps_ls = []
        with open(data_path, "r") as fp:
            data = json.load(fp)
            for gps_ls in data[0::3]:
                traces = []
                # trace_grids = []
                for gps in gps_ls:
                    gridx, gridy = utils.gps2grid(gps[0], gps[1])
                    traces.append(grid2traceid_dict[(gridx, gridy)] + 1)
                    # trace_grids.append([gridx, gridy])
                self.traces_ls.append(traces)
                # self.traces_gps_ls.append(trace_grids)
            self.roads_ls = data[1::3]
            self.traces_gps_ls = data[0::3]
            self.sampleIdx_ls = data[2::3]
        # self.traces_gps_ls = self.traces_gps_ls[:10000]
        # self.traces_ls = self.traces_ls[:10000]
        # self.roads_ls = self.roads_ls[:10000]
        # self.sampleIdx_ls = self.sampleIdx_ls[:10000]
        
        self.length = len(self.traces_ls)
        assert len(self.traces_ls) == len(self.roads_ls)

    def __getitem__(self, index):
        return self.traces_ls[index], self.roads_ls[index], self.traces_gps_ls[index], self.sampleIdx_ls[index]

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
        z.append(sample[2] + [[0,0]]*(max_tlen - len(sample[2])))
        w.append(sample[3] + [-1] * (max_tlen - len(sample[3])))
    f = torch.LongTensor
    return f(x), f(y), torch.FloatTensor(z), f(w), trace_lens, road_lens


if __name__ == "__main__":
    pass
