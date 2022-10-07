import sys
import random
import numpy as np
from datetime import datetime

# 随机下采样
def randomDownSampleBySize(sampleData: list, sampleRate : float, threshold: int) -> list:
    resData = list()
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
    return resData
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

    # 进行下采样
    sampleRate = [1 / 2, 1 / 4, 1 / 8]  # 对于clean data
    # sampleRate = [1/4, 1/8, 1/16]  # 对于clean data
    # sampleRate = [1/30, 1/60, 1/120]  # 对于未处理的数据，更改采样率
    for rate in sampleRate:
        tempRes = randomDownSampleBySize(finalLs, rate, threshold)
        # 将下采样的结果保存到文件
        with open('../sampleRes/threshold{}/sampledFullTrace{:.2f}.txt'.format(threshold,rate*100), 'w') as f:
            for i in tempRes:
                for j in i:
                    f.write(j)
    print("下采样完毕, 开始进行轨迹长度统计")

    # 统计不同threshold下的轨迹平均长度和轨迹个数
    for rate in sampleRate:
        filePath = '../sampleRes/threshold{}/sampledFullTrace{:.2f}.txt'.format(threshold,rate*100)
        trajList = readTrajFile(filePath, threshold)
        print("rate: {:.2f}%, threshold: {}, 轨迹个数={}".format(rate*100, threshold, len(trajList)))
        drawLengthDist(trajList)
        print("================")
