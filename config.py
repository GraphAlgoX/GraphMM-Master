import argparse


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_bsize", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200, help='max epochs')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--root_path",
                        type=str,
                        default="./data", help='the absolute path road.txt in')
    parser.add_argument(
        "--downsample_rate",
        type=float,
        default="0.5", help='downsample_rate in {0.5, 0.25, 0.125}')
    parser.add_argument("--emb_dim", type=int, default=256, help='embedding dimension')
    parser.add_argument("--layer", type=int, default=4, help='A^k num of layer neighbors')
    parser.add_argument("--wd", type=float, default=1e-8, help='Adamw weight decay')
    parser.add_argument("--dev_id", type=int, default=0, help='cuda id')
    parser.add_argument("--bi", action="store_true", help='use biGRU')
    parser.add_argument("--use_crf", action="store_true", help='use crf')
    parser.add_argument("--atten_flag", action="store_true", help='use attention in seq2seq')
    parser.add_argument("--tf_ratio", type=float, default=0.5, help='teacher forcing ratio')
    parser.add_argument("--drop_prob", type=float, default=0.5, help='dropout probability')
    parser.add_argument("--gamma", type=float, default=10000, help='penalty for unreachable')
    parser.add_argument("--topn", type=int, default=5, help='select topn in test mode')
    parser.add_argument("--neg_nums", type=int, default=800, help='select negetive sampling number')
    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    pass