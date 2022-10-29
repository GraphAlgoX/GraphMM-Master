
for idx in range(100):
    idx = '000000' + str(idx)
    if len(idx) < 8:
        idx = '0' + idx
    path = '/data/GeQian/g2s_2/map-matching-dataset/' + idx + '/'
    data_path = path + 'data/'
    with open(path+idx+'.arcs', 'r') as f:
        tmp_ls = f.readlines()
    print(len(tmp_ls))