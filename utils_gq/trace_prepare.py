import random
import numpy as np
import torch 
# utils是工具包
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from torch.utils.data import DataLoader # DataLoader需实例化，用于加载数据

data_path = '/data/GeQian/g2s_2/data_for_GMM-Master/data/'

class MyDataset(Dataset):   # 继承Dataset类
    def __init__(self, traces_ls, roads_ls): 
        # 把数据和标签拿出来
        self.traces_ls = traces_ls
        self.roads_ls = roads_ls
        
        assert len(self.traces_ls) == len(self.roads_ls)

        # 数据集的长度
        self.length = len(self.traces_ls)
        
    # 下面两个魔术方法比较好写，直接照着这个格式写就行了 
    def __getitem__(self, index): # 参数index必写
        return self.traces_ls[index], self.roads_ls[index]
    
    def __len__(self): 
        return self.length # 只需返回数据集的长度即可

# 实例化
# my_dataset = MyDataset() 
# train_loader = DataLoader(dataset=my_dataset, # 要传递的数据集
#                           batch_size=32, #一个小批量数据的大小是多少
#                           shuffle=True, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
#                           num_workers=0) # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误。



# 随机下采样
def randomDownSampleBySize(sampleData: list, sampleRate : float, threshold: int) -> list:
    resData = []
    pureData = []
    for i in range(len(sampleData)):
        trajList = sampleData[i]
        tempRes = [trajList[0], trajList[1]]  # 首节点
        for j in range(2, len(trajList)-1):
            if(random.random() <= sampleRate):
                tempRes.append(trajList[j])
        tempRes.append(trajList[-1])  # 尾节点
        # 长度过于短的忽略
        if(len(tempRes) < threshold): continue
        resData.append(tempRes)
        pureData.append(trajList)
    return resData, pureData
# 均匀下采样
def avgDownSampleBySize(sampleData: list, sampleGap : int=15, threshold: int=4) -> list:
    resData = list()
    for i in range(len(sampleData)):
        trajList = sampleData[i]
        tempRes = [trajList[0], trajList[1]]  # 首节点
        for j in range(2, len(trajList)):
            if (j % sampleGap == 0):
                tempRes.append(trajList[j])
        # 长度过于短的忽略
        if (len(tempRes) < threshold): continue
        resData.append(tempRes)
    return resData


# 统计轨迹平均长度
def drawLengthDist(finalLs: list):
    # 统计所有轨迹的平均长度
    avgLen = 0
    # 统计所有长度的分布
    allLenDic = dict()
    for i in range(len(finalLs)):
        temLen = len(finalLs[i]) - 1
        if(temLen in allLenDic.keys()): allLenDic[temLen] += 1
        else: allLenDic[temLen] = 1
        avgLen += temLen
    avgLen /= len(finalLs)
    print("轨迹平均长度={}".format(avgLen))

# 读取清洗后的数据
def readTrajFile(filePath, threshold=4):
    with open(filePath, 'r') as f:
        traj_list = f.readlines()
    finalLs = list()  # 用来保存所有轨迹
    tempLs = list()  # 用来保存单个轨迹
    for idx, sen in enumerate(traj_list):
        if sen[0] == '#':  # 表明是一条轨迹的开头
            if(idx != 0 and len(tempLs) >= threshold): finalLs.append(tempLs)
            tempLs = [sen]
        else:  # 增加轨迹点
            tempLs.append(sen)
    return finalLs


if __name__ == "__main__":
    
    # 设置阈值，将过短的轨迹删除掉
    threshold = 4

    # # 未清洗的未合并轨迹数据
    # old_full_trace_path = '/data/GeQian/g2s_2/preprocessed_pure/full_trace.txt'
    # # 清洗的未合并轨迹数据(clean data)
    # old_clean_full_trace_path = './clean_full_trace.txt'
    # 未清洗的合并的轨迹数据
    new_full_trace_path = '/data/GeQian/g2s_2/data_for_GMM-Master/data/pure_data/full_trace_new.txt'
    # 清洗的合并的轨迹数据
    new_clean_full_trace_path = '/data/GeQian/g2s_2/data_for_GMM-Master/data/pure_data/clean_full_trace_new.txt'


    # 从文件中读取数据
    finalLs = readTrajFile(new_full_trace_path, threshold)
    print("===========原数据长度===========")
    drawLengthDist(finalLs)
    print("===========   END   ===========\n")

    print("读取数据完毕, 先进行均匀下采样")
    finalLs = avgDownSampleBySize(finalLs, 15, 4)
    print("===========均匀采样后数据长度===========")
    drawLengthDist(finalLs)
    print("===========   END   ===========\n")

    with open(data_path+'pure_data/full_trace_new.txt','w') as f:
        for i in finalLs:
            for j in i:
                f.write(j)

    # sampleRate = 0.5  # 对于clean data

    # downsampleData, pureData = randomDownSampleBySize(finalLs, sampleRate, threshold)

    
    # print("下采样完毕, 开始进行轨迹长度统计")


    # traces_ls = []
    # roads_ls = []
    # for downdata, puredata in zip(downsampleData, pureData):
    #     traces = []
    #     roads = []
    #     for i in downdata:
    #         if i[0] == '#':
    #             continue
    #         il = i.split(',')
    #         lat = float(il[1])
    #         lng = float(il[2])
    #         # road_id = int(i[3])
    #         traces.append((lat, lng))
    #         # roads.append(road_id)
    #     for i in puredata:
    #         if i[0] == '#':
    #             continue
    #         roads.append(int(i.split(',')[3]))
    #     traces_ls.append(traces)
    #     roads_ls.append(roads)
    

    # dataset = MyDataset(traces_ls=traces_ls, roads_ls=roads_ls)

    # dlens = dataset.length
    # train_lens, val_lens = int(0.7*dlens), int(0.2*dlens)
    # test_lens = dlens - train_lens - val_lens
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, 
    #     [train_lens, val_lens, test_lens])
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    # print('finished ')
    # for trace, road in train_loader:
    #     print(f'trace={trace}')
    #     print(f'road={road}')
    #     break

    
