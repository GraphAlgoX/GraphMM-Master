import os
import random
import json
import math


def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def randomDownSampleBySize(sampleData: list, sampleRate: float,
                           threshold: int) -> list:
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
        # ignore short traj
        if (len(tempRes) < threshold):
            continue
        resData.append(tempRes)
        pureData.append(trajList)
        resIdx.append(tmpIdx)
    return resData, pureData, resIdx


def uniformDownSampleBySize(sampleData: list,
                            sampleGap: int = 15,
                            threshold: int = 4) -> list:
    """
    uniform sampling
    """
    resData = list()
    for i in range(len(sampleData)):
        trajList = sampleData[i]
        tempRes = [trajList[0], trajList[1]]  # 首节点
        for j in range(2, len(trajList)):
            if (j % sampleGap == 0):
                tempRes.append(trajList[j])
        # 长度过于短的忽略
        if (len(tempRes) < threshold):
            continue
        resData.append(tempRes)
    return resData


class DataProcess():
    def __init__(self, traj_input_path, output_dir, threshold,
                 sample_rate, max_road_len=25, min_road_len=15) -> None:
        self.traj_input_path = traj_input_path
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.max_road_len = max_road_len
        self.min_road_len = min_road_len
        beginLs = self.readTrajFile(traj_input_path)
        # self.max_road_len = self.getSuitableCut(beginLs)
        self.finalLs = self.cutData(beginLs)
        # self.write_cut_data()
        # self.traces_ls, self.roads_ls, self.downsampleIdx = self.sampling()
        # self.splitData(output_dir)

    def readTrajFile(self, filePath):
        full_lens = 0
        with open(filePath, 'r') as f:
            traj_list = f.readlines()
        finalLs = list()  # 用来保存所有轨迹
        tempLs = list()  # 用来保存单个轨迹
        for idx, sen in enumerate(traj_list):
            if sen[0] == '#':  # 表明是一条轨迹的开头
                if (idx != 0 and len(tempLs) >= self.threshold):
                    full_lens += len(tempLs)-1
                    finalLs.append(tempLs)
                tempLs = [sen]
            else:  # 增加轨迹点
                tempLs.append(sen)
        return finalLs

    def getSuitableCut(self, beginLs):
        num_of_trace = len(beginLs)
        trace_lens_ls = [len(i)-1 for i in beginLs]
        sum_lens = sum(trace_lens_ls)
        avg_lens = (0.0+sum_lens)/num_of_trace
        d = 0.0
        for i in trace_lens_ls:
            d += (i-avg_lens)**2
        sd = math.sqrt(d/num_of_trace)
        suitable_cut = avg_lens + 2*sd
        print(f'sum_lens = {sum_lens}, avg_lens = {avg_lens}, sd = {sd}, suitable_cut = {suitable_cut}')
        return suitable_cut

    def sampling(self):
        """
        down sampling
        """
        downsampleData, pureData, downsampleIdx = randomDownSampleBySize(
            self.finalLs, self.sample_rate, self.threshold)
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
        return traces_ls, roads_ls, downsampleIdx

    def cutData(self, beginLs): 
        # each trace [min_lens+1, max_lens+min_lens+1) 
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

        with open('/data/GeQian/g2s2hx/final_trace_after_cut.txt', 'w') as f:
            for traces in finalLs:
                for i in traces:
                    f.write(i)

        return finalLs

    def splitData(self, output_dir, train_rate=0.7, val_rate=0.1):
        """
        split original data to train, valid and test datasets
        """
        create_dir(output_dir)
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
        for i in range(num_sample):
            if i in train_idxs:
                trainset.extend([self.traces_ls[i], self.roads_ls[i], self.downsampleIdx[i]])
            elif i in val_idxs:
                valset.extend([self.traces_ls[i], self.roads_ls[i], self.downsampleIdx[i]])
            else:
                testset.extend([self.traces_ls[i], self.roads_ls[i], self.downsampleIdx[i]])

        with open(os.path.join(train_data_dir, "train.json"), 'w') as fp:
            json.dump(trainset, fp)

        with open(os.path.join(val_data_dir, "val.json"), 'w') as fp:
            json.dump(valset, fp)

        with open(os.path.join(test_data_dir, "test.json"), 'w') as fp:
            json.dump(testset, fp)


if __name__ == "__main__":
    DataProcess(traj_input_path='/data/GeQian/g2s_2/data_for_GMM-Master/data/pure_data/full_trace_new.txt', \
        output_dir='/data/GeQian/g2s_2/data_for_GMM-Master/data/', threshold=4, sample_rate=0.5)
    pass

