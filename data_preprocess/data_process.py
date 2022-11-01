import os
import random
import json
import math
import sys
from utils import gps2grid, grid2gps, create_dir, get_border

MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG = get_border('../data/road.txt')
GRID_SIZE = 50

def randomDownSampleBySize(sampleData: list, sampleRate: float) -> list:
    """
        randomly sampling
    """
    resData, pureData, resIdx = [], [], []
    for i in range(len(sampleData)):
        trajList = sampleData[i]
        tempRes = [trajList[0], trajList[1]]  # 首节点
        tmpIdx = [0]
        for j in range(2, len(trajList) - 1):
            if (random.random() <= sampleRate):
                tempRes.append(trajList[j])
                tmpIdx.append(j-1)
        tempRes.append(trajList[-1])  # 尾节点
        tmpIdx.append(len(trajList) - 2)
        resData.append(tempRes)
        pureData.append(trajList)
        resIdx.append(tmpIdx)
    return resData, pureData, resIdx


class DataProcess():
    def __init__(self, traj_input_path, output_dir, 
                 sample_rate, max_road_len=25, min_road_len=15) -> None:
        self.traj_input_path = traj_input_path
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.max_road_len = max_road_len
        self.min_road_len = min_road_len
        beginLs = self.readTrajFile(traj_input_path)
        self.finalLs = self.cutData(beginLs)
        self.traces_ls, self.roads_ls, self.downsampleIdx, downSampleData = self.sampling()
        self.splitData(output_dir)
        with open(self.output_dir + 'data_split/downsample_trace.txt', 'w') as f:
            for traces in downSampleData:
                for trace in traces:
                    f.write(trace)

    def readTrajFile(self, filePath):
        """
            read trace.txt
        """
        with open(filePath, 'r') as f:
            traj_list = f.readlines()
        finalLs = list()  # 用来保存所有轨迹
        tempLs = list()  # 用来保存单个轨迹
        for idx, sen in enumerate(traj_list):
            if sen[0] == '#':  # 表明是一条轨迹的开头
                if idx != 0:
                    finalLs.append(tempLs)
                tempLs = [sen]
            else:  # 增加轨迹点
                tempLs.append(sen)
        finalLs.append(tempLs)
        return finalLs

    def sampling(self):
        """
            down sampling
        """
        downsampleData, pureData, downsampleIdx = randomDownSampleBySize(self.finalLs, self.sample_rate)
        traces_ls, roads_ls = [], []
        for downdata, puredata in zip(downsampleData, pureData):
            traces, roads = [], []
            for i in downdata:
                if i[0] == '#':
                    continue
                il = i.split(',')
                lat, lng = float(il[1]), float(il[2])
                traces.append((lat, lng))
            for i in puredata:
                if i[0] == '#':
                    continue
                roads.append(int(i.split(',')[3]))
            traces_ls.append(traces)
            roads_ls.append(roads)
        return traces_ls, roads_ls, downsampleIdx, downsampleData

    def cutData(self, beginLs): 
        """
            ensure each trace's length in [min_lens+1, max_lens+min_lens+1) 
        """
        finalLs = []
        for traces in beginLs:
            assert traces[0][0] == '#'
            title = traces[0]
            traces = traces[1:]
            lens = len(traces)
            if lens < self.min_road_len:
                continue
            if lens < self.max_road_len and lens >= self.min_road_len:
                finalLs.append([title]+traces)
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
                    tmp_ls = [title] + traces[i*self.max_road_len:(i+1) * self.max_road_len]
                    finalLs.append(tmp_ls)

                assert(lens - (int_cutnum-1)*self.max_road_len < self.max_road_len + self.min_road_len)
                latLS = [title] + traces[(int_cutnum-1) * self.max_road_len:]

                finalLs.append(latLS)

        for i in finalLs:
            assert(len(i) >= 16 and len(i) <= 40)
        return finalLs

    def splitData(self, output_dir, train_rate=0.7, val_rate=0.1):
        """
            split original data to train, valid and test datasets
        """
        create_dir(output_dir)
        create_dir(output_dir + 'data_split/')
        train_data_dir = output_dir + 'train_data/'
        create_dir(train_data_dir)
        val_data_dir = output_dir + 'val_data/'
        create_dir(val_data_dir)
        test_data_dir = output_dir + 'test_data/'
        create_dir(test_data_dir)
        num_sample = len(self.traces_ls)
        train_size, val_size = int(num_sample * train_rate), int(num_sample *
                                                                 val_rate)
        idxs = list(range(num_sample))
        random.shuffle(idxs)
        train_idxs = idxs[:train_size]
        val_idxs = idxs[train_size:train_size + val_size]
        trainset, valset, testset = [], [], []

        train_trace = []
        val_trace = []
        test_trace = []
        for i in range(num_sample):
            if i in train_idxs:
                trainset.extend([self.traces_ls[i], self.roads_ls[i], self.downsampleIdx[i]])
                train_trace += [self.finalLs[i]]
            elif i in val_idxs:
                valset.extend([self.traces_ls[i], self.roads_ls[i], self.downsampleIdx[i]])
                val_trace += [self.finalLs[i]]
            else:
                testset.extend([self.traces_ls[i], self.roads_ls[i], self.downsampleIdx[i]])
                test_trace += [self.finalLs[i]]

        with open(os.path.join(train_data_dir, "train.json"), 'w') as fp:
            json.dump(trainset, fp)

        with open(os.path.join(val_data_dir, "val.json"), 'w') as fp:
            json.dump(valset, fp)

        with open(os.path.join(test_data_dir, "test.json"), 'w') as fp:
            json.dump(testset, fp)

        all_trace = [train_trace, val_trace, test_trace]
        all_trace_name = ['train_trace.txt', 'val_trace.txt', 'test_trace.txt']
        for i in range(3):
            tmptrace = all_trace[i]
            path = output_dir + 'data_split/' + all_trace_name[i]
            with open(path, 'w') as f:
                for traces in tmptrace:
                    for trace in traces:
                        f.write(trace)




if __name__ == "__main__":
    downsample_rate = sys.argv[1]
    path = '../data/'
    data_path = path + 'data' + downsample_rate + '/'
    DataProcess(traj_input_path=path+'trace.txt', \
        output_dir=data_path, sample_rate=float(downsample_rate))
    pass

