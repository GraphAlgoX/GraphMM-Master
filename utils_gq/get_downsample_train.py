import json
import datetime
def readTrajFile(filePath):
    full_lens = 0
    with open(filePath, 'r') as f:
        traj_list = f.readlines()
    finalLs = list()  # 用来保存所有轨迹
    tempLs = list()  # 用来保存单个轨迹
    for idx, sen in enumerate(traj_list):
        if sen[0] == '#':  # 表明是一条轨迹的开头
            if idx != 0:
                full_lens += len(tempLs)-1
                finalLs.append(tempLs)
            tempLs = [sen]
        else:  # 增加轨迹点
            tempLs.append(sen)
    
    finalLs.append(tempLs)
    return finalLs

def readTrajFile(filePath):
    full_lens = 0
    with open(filePath, 'r') as f:
        traj_list = f.readlines()
    finalLs = list()  # 用来保存所有轨迹
    tempLs = list()  # 用来保存单个轨迹
    for idx, sen in enumerate(traj_list):
        if sen[0] == '#':  # 表明是一条轨迹的开头
            
            if idx != 0:
                full_lens += len(tempLs)-1
                finalLs.append(tempLs)
            tempLs = [sen]
        else:  # 增加轨迹点
            tempLs.append(sen)
    
    finalLs.append(tempLs)
    return finalLs

def get_train_dict(finalLs):
    train_dict = {}
    for traces in finalLs: 
        trace_id = traces[0].strip().split(',')[1]
        timestamp = traces[1].strip().split(',')[0]
        train_dict[(trace_id, timestamp)] = traces
    return train_dict


path = '/data/GeQian/g2s_2/gmm-data0.5/data/data_for_mtraj/'

train_trace_ls = readTrajFile(path+'train_trace.txt')

full_downs_ls = readTrajFile(path+'downsample_trace.txt')

train_dict = get_train_dict(train_trace_ls)


with open(path + 'train_downsample_trace.txt', 'w') as f:
    for traces in full_downs_ls:
        trace_id = traces[0].strip().split(',')[1]
        timestamp = traces[1].strip().split(',')[0]
        if (trace_id, timestamp) in train_dict.keys():
            for trace in traces:
                f.write(trace)
    print('finished')