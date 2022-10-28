def cal_id_acc(predict, target, trg_len):
    """
    predict, target: (batch_size, seq_len)
    trg_len: the real length of every traj
    """
    batch_bingo, batch_acc = 0., []
    batch_size = predict.size(0)
    for i in range(batch_size):
        cur_tlen = trg_len[i]
        cur_pred = predict[i, :cur_tlen]
        cur_true = target[i, :cur_tlen]
        bingo = (cur_pred == cur_true).sum()
        batch_bingo += bingo
        batch_acc.append((bingo / cur_tlen).item())
    return batch_bingo, batch_acc