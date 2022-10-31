import torch
from config import get_params
from model.gmm import GMM
from utils_gq.data_loader import MyDataset, padding
from torch.utils.data import DataLoader
from graph_data import GraphData
from metrics import cal_id_acc
from tqdm import tqdm
import os.path as osp
from config import get_params


def evaluate(model, eval_iter, device, gdata, tf_ratio, use_crf):
    model.eval()
    global_acc, global_avg_lcs = [], []
    global_tnums = 0.
    with torch.no_grad():
        for data in tqdm(eval_iter):
            grid_traces = data[0].to(device)
            tgt_roads = data[1]
            traces_gps = data[2].to(device)
            sample_Idx = data[3].to(device)
            traces_lens = data[4]
            road_lens = data[5]
            infer_seq = model.infer(grid_traces=grid_traces,
                                    traces_gps=traces_gps,
                                    traces_lens=traces_lens,
                                    road_lens=road_lens,
                                    gdata=gdata,
                                    sample_Idx=sample_Idx,
                                    tf_ratio=tf_ratio)
            if use_crf:
                infer_seq = torch.tensor(infer_seq)
            else:
                infer_seq = infer_seq.argmax(dim=-1).detach().cpu()
            batch_acc, batch_avg_lcs = cal_id_acc(infer_seq, tgt_roads, road_lens)
            global_acc.extend(batch_acc)
            global_avg_lcs.extend(batch_avg_lcs)
            global_tnums += tgt_roads.size(0)
    acc_t = sum(global_acc) / global_tnums
    avg_lcs = sum(global_avg_lcs) / global_tnums
    return acc_t, avg_lcs

args = vars(get_params())
ckpt_path = "/data/LuoWei/Code/ckpt1w/bz32_lr0.0001_ep300_edim256_dp0.5_tf0.5_tn5_ng800_crfTrue_best2.pt"
root_path = osp.join(args['parent_path'], args['data_dir'])
testset = MyDataset(root_path, "test")
test_iter = DataLoader(dataset=testset,
                        batch_size=args['eval_bsize'],
                        collate_fn=padding)
print("Loading Dataset Done!!!")
device = torch.device(f"cuda:{args['dev_id']}" if torch.cuda.is_available() else "cpu")
gdata = GraphData(root_path=root_path,
                    layer=args['layer'],
                    gamma=args['gamma'],
                    device=device)
print('get graph extra data finished!')
model = GMM(emb_dim=args['emb_dim'],
            target_size=gdata.num_roads,
            topn=args['topn'],
            neg_nums=args['neg_nums'],
            device=device,
            use_crf=args['use_crf'],
            bi=args['bi'],
            atten_flag=args['atten_flag'],
            drop_prob=args['drop_prob'])
model.load_state_dict(torch.load(ckpt_path))
model = model.to(device)
print("Loading model Done!!!")
acc_t, avg_lcs = evaluate(model, test_iter, device, gdata, 0., args['use_crf'])
print(f"testset: acc(T)({acc_t:.4f}) acg_lcs({avg_lcs:.4f})")

