import os
import random
import json
import math
import sys
import copy

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

class DataProcess():
    def __init__(self, track_path, road_path, output_dir, max_road_len=25, min_road_len=15) -> None:
        self.track_path = track_path
        self.road_path = road_path
        with open(track_path, 'r') as f:
            track_ls = f.readlines()
        with open(road_path, 'r') as f:
            road_ls = f.readlines()
        self.output_dir = output_dir
        self.max_road_len = max_road_len
        self.min_road_len = min_road_len

        self.track_ls, self.road_ls = self.cutData(track_ls, road_ls)
        self.traces_ls, self.roads_ls, self.downsampleIdx = self.sampling()

    def cutData(self, traces, roads): 
        trace_ls = []
        road_ls = []
        lens = len(traces)
        if lens < self.max_road_len and lens >= self.min_road_len:
            trace_ls.append(traces)
        else:
            cutnum = lens/self.max_road_len
            int_cutnum = int(cutnum)
            lstnum = lens - int_cutnum*self.max_road_len
            if lens % self.max_road_len != 0:
                int_cutnum += 1
            else:
                lstnum = self.max_road_len
            if lstnum < self.min_road_len:
                int_cutnum -= 1

            for i in range(int_cutnum - 1):
                tmp_ls = traces[i*self.max_road_len:(i+1) * self.max_road_len]
                trace_ls.append(tmp_ls)

            assert(lens - (int_cutnum-1)*self.max_road_len < self.max_road_len + self.min_road_len)
            latLS = traces[(int_cutnum-1) * self.max_road_len:]
            trace_ls.append(latLS)

        lst_idx = 0
        for i in trace_ls:
            lens = len(i)
            road_ls.append(roads[lst_idx:lst_idx+lens])
            lst_idx += lens
            assert(len(i) >= 15 and len(i) <= 40)

        return trace_ls, road_ls


    def sampling(self):
        """
        down sampling
        """
        # downsampleData = copy.deepcopy(self.finalLs)
        downsampleIdx = []
        traces_ls, roads_ls = [], []
        for trace in self.track_ls:
            traces = []
            for i in trace:
                i = i.strip()
                il = i.split('\t')
                lat, lng = float(il[1]), float(il[0])
                traces.append((lat, lng))
            traces_ls.append(traces)
        
        for road in self.road_ls:
            roads = [int(i.strip()) for i in road]
            roads_ls.append(roads)
            lens = len(road)
            downsampleIdx.append(list(range(lens)))
        
        testset = []
        # 
            # test_trace += [self.finalLs[i]]
        num = len(traces_ls)
        assert(len(traces_ls) == len(roads_ls))
        assert(len(downsampleIdx) == len(roads_ls))
        for i in range(num):
            testset.extend([traces_ls[i], roads_ls[i], downsampleIdx[i]])
        test_data_dir = self.output_dir + 'test_data/'
        create_dir(test_data_dir)
        with open(os.path.join(test_data_dir, "test.json"), 'w') as fp:
            json.dump(testset, fp)

        return traces_ls, roads_ls, downsampleIdx


if __name__ == "__main__":
    idx = str(sys.argv[1])
    idx = '000000' + idx
    if len(idx) < 8:
        idx = '0' + idx
    path = '/data/GeQian/g2s_2/map-matching-dataset/' + idx + '/'
    data_path = path + 'data/'
    DataProcess(track_path=path+idx+'.track', road_path=path+idx+'.new_route', output_dir=data_path)
    pass

