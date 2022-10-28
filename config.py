import argparse


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_bsize", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--parent_path",
                        type=str,
                        default="/data/LuoWei/Code")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="gmm-data0.5",
        choices=["gmm-data0.5", "gmm-data0.25", "gmm-data0.125"])
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--wd", type=float, default=1e-8)
    parser.add_argument("--dev_id", type=int, default=0)
    parser.add_argument("--bi", action="store_true")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--atten_flag", action="store_true")
    parser.add_argument("--tf_ratio", type=float, default=0.5)
    parser.add_argument("--drop_prob", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=10000)
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--neg_nums", type=int, default=500)
    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    pass