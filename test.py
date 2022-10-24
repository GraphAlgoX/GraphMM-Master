import nni
import numpy as np
import torch
import torch.optim as optim
from config import get_params
from nni.utils import merge_parameter
from model.gmm import GMM
from utils_gq.data_loader import MyDataset, padding
from torch.utils.data import DataLoader
from graph_data import GraphData
from metrics_calculate import cal_id_acc
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import os.path as osp
from config import get_params


def evaluate(model, eval_iter, device, gdata, tf_ratio):
    model.eval()
    eval_acc_sum, eval_r_sum, eval_p_sum = 0., 0., 0.
    count = 0
    with torch.no_grad():
        for data in tqdm(eval_iter):
            grid_traces = data[0].to(device)
            tgt_roads = data[1]
            traces_gps = data[2].to(device)
            sample_Idx = data[3].to(device)
            traces_lens = data[4]
            road_lens = data[5]
            _, infer_seq = model.infer(grid_traces=grid_traces,
                                       traces_gps=traces_gps,
                                       traces_lens=traces_lens,
                                       road_lens=road_lens,
                                       gdata=gdata,
                                       sample_Idx=sample_Idx,
                                       tf_ratio=tf_ratio)
            tgt_roads = tgt_roads.flatten().numpy()
            mask = (tgt_roads != -1)
            # infer_seq = infer_seq.detach().cpu()
            # _, indices = torch.topk(infer_seq, dim=-1, k=args['topn'])
            # indices = indices.reshape(-1, args['topn'])
            # indices = indices[mask]
            # tgt_roads = tgt_roads[mask]
            # bingo = 0
            # for gt, topk in zip(tgt_roads, indices):
            #     if gt in topk:
            #         bingo += 1
            # acc = bingo / tgt_roads.shape[0]
            # infer_seq = infer_seq.argmax(dim=-1).detach().cpu().numpy().flatten()
            infer_seq = np.array(infer_seq).flatten()
            acc = accuracy_score(infer_seq[mask], tgt_roads[mask])
            # acc, recall, precision = cal_id_acc(infer_seq, tgt_roads,
            #                                     road_lens)
            eval_acc_sum += acc
            eval_r_sum += 0
            eval_p_sum += 0
            # eval_r_sum += recall
            # eval_p_sum += precision
            count += 1
            # exit(0)
    return eval_acc_sum / count, eval_r_sum / count, eval_p_sum / count

args = vars(get_params())
ckpt_path = "/data/LuoWei/Code/ckpt/bz32_lr0.0001_ep200_edim256_dp0.5_tf0.5_tn30_ng800_best.pt"
root_path = osp.join(args['parent_path'], 'gmm-data')
testset = MyDataset(root_path, "test")
test_iter = DataLoader(dataset=testset,
                        batch_size=args['eval_bsize'],
                        collate_fn=padding)
print("Loading Dataset Done!!!")
# args['dev_id'] = 1 if args['use_gcn'] else 0
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
            atten_flag=args['atten_flag'],
            drop_prob=args['drop_prob'])
model.load_state_dict(torch.load(ckpt_path))
model = model.to(device)
best_acc = 0.
print("Loading model Done!!!")
# loss_fn = nn.NLLLoss()
test_avg_acc, test_avg_r, test_avg_p = evaluate(model, test_iter, device, gdata, 0.)
nni.report_final_result(test_avg_acc)
print(f"testset: acc({test_avg_acc}) recall({test_avg_r}) precision({test_avg_p})")

