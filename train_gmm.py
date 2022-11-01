import nni
import numpy as np
import torch
import torch.optim as optim
from config import get_params
from nni.utils import merge_parameter
from model.gmm import GMM
from data_loader import MyDataset, padding
from torch.utils.data import DataLoader
from graph_data import GraphData
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score
import os.path as osp
from copy import deepcopy
from data_preprocess.utils import create_dir

def train(model, train_iter, optimizer, device, gdata, args):
    model.train()
    train_l_sum, count = 0., 0
    for data in train_iter:
        grid_traces = data[0].to(device)
        tgt_roads = data[1].to(device)
        traces_gps = data[2].to(device)
        sample_Idx = data[3].to(device)
        traces_lens, road_lens = data[4], data[5]
        loss = model(grid_traces=grid_traces,
                     traces_gps=traces_gps,
                     traces_lens=traces_lens,
                     road_lens=road_lens,
                     tgt_roads=tgt_roads,
                     gdata=gdata,
                     sample_Idx=sample_Idx,
                     tf_ratio=args['tf_ratio'])
        train_l_sum += loss.item()
        count += 1
        if count % 5 == 0:
            print(f"Iteration {count}: train_loss {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    return train_l_sum / count


def evaluate(model, eval_iter, device, gdata, use_crf):
    model.eval()
    eval_acc_sum, count = 0., 0
    with torch.no_grad():
        for data in tqdm(eval_iter):
            grid_traces = data[0].to(device)
            tgt_roads = data[1]
            traces_gps = data[2].to(device)
            sample_Idx = data[3].to(device)
            traces_lens, road_lens = data[4], data[5]
            infer_seq = model.infer(grid_traces=grid_traces,
                                    traces_gps=traces_gps,
                                    traces_lens=traces_lens,
                                    road_lens=road_lens,
                                    gdata=gdata,
                                    sample_Idx=sample_Idx,
                                    tf_ratio=0.)
            if use_crf:
                infer_seq = np.array(infer_seq).flatten()
            else:
                infer_seq = infer_seq.argmax(dim=-1).detach().cpu().numpy().flatten()
            tgt_roads = tgt_roads.flatten().numpy()
            mask = (tgt_roads != -1)
            acc = accuracy_score(infer_seq[mask], tgt_roads[mask])
            eval_acc_sum += acc
            count += 1
    return eval_acc_sum / count


def main(args):
    create_dir(f"{args['root_path']}/ckpt/")
    save_path = "{}/ckpt/bz{}_lr{}_ep{}_edim{}_dp{}_tf{}_tn{}_ng{}_crf{}_wd{}_best.pt".format(
        args['root_path'], args['batch_size'], args['lr'], args['epochs'],
        args['emb_dim'], args['drop_prob'], args['tf_ratio'], args['topn'],
        args['neg_nums'], args['use_crf'], args['wd'])
    root_path = args['root_path']

    data_path = osp.join(args['root_path'], 'data'+str(args['downsample_rate']) + '/')
    trainset = MyDataset(root_path=root_path, path=data_path, name="train")
    valset = MyDataset(root_path=root_path, path=data_path, name="val")
    testset = MyDataset(root_path=root_path, path=data_path, name="test")
    train_iter = DataLoader(dataset=trainset,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=padding)
    val_iter = DataLoader(dataset=valset,
                          batch_size=args['eval_bsize'],
                          collate_fn=padding)
    test_iter = DataLoader(dataset=testset,
                           batch_size=args['eval_bsize'],
                           collate_fn=padding)
    print("loading dataset finished!")
    device = torch.device(f"cuda:{args['dev_id']}" if torch.cuda.is_available() else "cpu")
    gdata = GraphData(root_path=root_path,
                      data_path=data_path,
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
    model = model.to(device)
    best_acc, best_model = 0., None
    print("loading model finished!")
    optimizer = optim.AdamW(params=model.parameters(),
                            lr=args['lr'],
                            weight_decay=args['wd'])
    # start training
    for e in range(args['epochs']):
        print(f"================Epoch: {e + 1}================")
        train_avg_loss = train(model, train_iter, optimizer, device, gdata, args)
        val_acc = evaluate(model, val_iter, device, gdata, args['use_crf'])
        # choose model based on val_acc
        if best_acc < val_acc:
            best_model = deepcopy(model)
            best_acc = val_acc
        print("Epoch {}: train_avg_loss {} val_avg_acc: {}".format(e + 1, train_avg_loss, val_acc))
        nni.report_intermediate_result(val_acc)


    test_acc = evaluate(best_model, test_iter, device, gdata, args['use_crf'])
    nni.report_final_result(test_acc)
    print(f"test_avg_acc: {test_acc:.4f}")
    torch.save(best_model.state_dict(), save_path)


if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter()
        if tuner_params.get('tf_ratio') and tuner_params['tf_ratio'] == 0:
            tuner_params['tf_ratio'] = 0.0
        if tuner_params.get('drop_prob') and tuner_params['drop_prob'] == 0:
            tuner_params['drop_prob'] = 0.0
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        raise
