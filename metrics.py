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


def cal_id_acc(predict, target, trg_len):
    """
    predict, target: (batch_size, seq_len)
    trg_len: the real length of every traj
    """
    batch_acc, batch_avg_lcs = [], []
    batch_size = predict.size(0)
    for i in range(batch_size):
        cur_tlen = trg_len[i]
        cur_pred = predict[i, :cur_tlen]
        cur_true = target[i, :cur_tlen]
        bingo = (cur_pred == cur_true).sum()
        cur_lcs = lcs(cur_pred, cur_true)
        batch_acc.append((bingo / cur_tlen).item())
        batch_avg_lcs.append(len(cur_lcs) / cur_tlen)
    return batch_acc, batch_avg_lcs