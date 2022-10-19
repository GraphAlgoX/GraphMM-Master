import argparse


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--parent_path",
                        type=str,
                        default="/data/LuoWei/Code")
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--wd", type=float, default=1e-8)
    parser.add_argument("--dev_id", type=int, default=0)
    parser.add_argument("--use_gcn", type=int, default=1)
    parser.add_argument("--atten_flag", type=int, default=1)
    parser.add_argument("--tf_ratio", type=float, default=0.8)
    parser.add_argument("--drop_prob", type=float, default=0.5)
    parser.add_argument("--lambda", type=float, default=0.1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    pass