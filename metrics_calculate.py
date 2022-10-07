def memoize(fn):
    '''
    Return a memoized version of the input function.
    The returned function caches the results of previous calls.
    Useful if a function call is expensive, and the function 
    is called repeatedly with the same arguments.
    '''
    cache = dict()

    def wrapped(*v):
        key = tuple(v)  # tuples are hashable, and can be used as dict keys
        if key not in cache:
            cache[key] = fn(*v)
        return cache[key]

    return wrapped


def lcs(xs, ys):
    '''
    Return the longest subsequence common to xs and ys.
    Example
    >>> lcs("HUMAN", "CHIMPANZEE")
    ['H', 'M', 'A', 'N']
    '''
    @memoize
    def lcs_(i, j):
        if i and j:
            xe, ye = xs[i - 1], ys[j - 1]
            if xe == ye:
                return lcs_(i - 1, j - 1) + [xe]
            else:
                return max(lcs_(i, j - 1), lcs_(i - 1, j), key=len)
        else:
            return []

    return lcs_(len(xs), len(ys))


def shrink_seq(seq):
    """remove repeated ids"""
    s0 = seq[0]
    new_seq = [s0]
    for s in seq[1:]:
        if s == s0:
            continue
        else:
            new_seq.append(s)
        s0 = s

    return new_seq


def cal_id_acc(predict, target, trg_len):

    # predict [batch size, seq len]
    # target [batch size, seq len]
    bs = predict.size(0)

    correct_id_num = 0
    ttl_trg_id_num = 0
    ttl_pre_id_num = 0
    ttl = 0
    cnt = 0
    for bs_i in range(bs):
        pre_ids = []
        trg_ids = []

        for len_i in range(trg_len[bs_i]):
            pre_id = predict[bs_i][len_i]
            trg_id = target[bs_i][len_i]
            pre_ids.append(pre_id)
            trg_ids.append(trg_id)
            if pre_id == trg_id:
                cnt += 1
            ttl += 1

        # compute average rid accuracy
        shr_trg_ids = shrink_seq(trg_ids)
        shr_pre_ids = shrink_seq(pre_ids)
        correct_id_num += len(lcs(shr_trg_ids, shr_pre_ids))
        ttl_trg_id_num += len(shr_trg_ids)
        ttl_pre_id_num += len(shr_pre_ids)

    rid_acc = cnt / ttl
    rid_recall = correct_id_num / ttl_trg_id_num
    rid_precision = correct_id_num / ttl_pre_id_num
    return rid_acc, rid_recall, rid_precision


if __name__ == "__main__":
    pass
