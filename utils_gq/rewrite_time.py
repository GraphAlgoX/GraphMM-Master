import datetime
# print (+datetime.timedelta(days=-1)).strftime("%Y-%m-%d %H:%M:%S")

# date_string = '2022/03/31 01:25:34'
# lt = datetime.datetime.strptime(date_string, '%Y/%m/%d %H:%M:%S')
# nt = (lt + datetime.timedelta(seconds=15)).strftime("%Y/%m/%d %H:%M:%S")
# print(nt, type(nt))

def readTrajFile(filePath):
    with open(filePath, 'r') as f:
        traj_list = f.readlines()
    finalLs = []
    tempLs = []
    for idx, sen in enumerate(traj_list):
        if sen[0] == '#':  # 表明是一条轨迹的开头
            if(idx != 0): finalLs.append(tempLs)
            tempLs = [sen]
        else:  # 增加轨迹点
            tempLs.append(sen)
    return finalLs



finalLS = readTrajFile('/data/GeQian/g2s_2/gmm_data/data/pure_data/full_trace_new.txt')
with open('/data/GeQian/g2s_2/gmm_data/data/pure_data/full_trace_new.txt', 'w') as f:
    for ls in finalLS:
        assert(len(ls) > 10)
        if len(ls) < 5:
            continue
        add_flag = True
        f.write(ls[0])
        # print(ls[0])
        lst_time = ls[1][:19]
        f.write(ls[1])
        # print(ls[1])
        lt = datetime.datetime.strptime(lst_time, '%Y/%m/%d %H:%M:%S')
        for i in ls[2:]:
            nt = (lt + datetime.timedelta(seconds=15)).strftime("%Y/%m/%d %H:%M:%S")
            tmpst = nt+i[19:]
            f.write(tmpst)
            lt = datetime.datetime.strptime(nt, '%Y/%m/%d %H:%M:%S')
            # print(tmpst)
        # exit(0)

        


