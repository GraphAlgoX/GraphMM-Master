import nni
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
# from torchviz import make_dot


def train(model, train_iter, loss_fn, optimizer, device, gdata, tf_ratio):
    model.train()
    train_l_sum, count = 0., 0

    for data in train_iter:
        # model.init_cache()
        # print('finish init!')
        grid_traces = data[0].to(device)
        tgt_roads = data[1].to(device)
        traces_gps = data[2].to(device)
        traces_lens = torch.tensor(data[3])
        road_lens = torch.tensor(data[4])
        y_pred = model(grid_traces=grid_traces,
                    traces_gps=traces_gps,
                    traces_lens=traces_lens,
                    road_lens=road_lens,
                    tgt_roads=tgt_roads,
                    gdata=gdata,
                    tf_ratio=tf_ratio)
        # g = make_dot(y_pred, params=dict(model.named_parameters()))
        # g.render('gmm', view=False)
        # print(y_pred.shape, tgt_roads.shape)
        mask = (tgt_roads.view(-1) != -1)
        loss = loss_fn(y_pred.view(-1, y_pred.shape[-1])[mask], tgt_roads.view(-1)[mask])
        train_l_sum += loss.item()
        count += 1
        if count % 1 == 0:
            print(f"Iteration {count}: train_loss {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    return train_l_sum / count


def evaluate(model, eval_iter, device, gdata, tf_ratio):
    model.eval()
    eval_acc_sum, eval_r_sum, eval_p_sum = 0., 0., 0.
    count = 0
    with torch.no_grad():
        for data in tqdm(eval_iter):
            grid_traces = data[0].to(device)
            tgt_roads = data[1]
            traces_gps = data[2].to(device)
            traces_lens = data[3]
            road_lens = data[4]
            _, infer_seq = model.infer(grid_traces=grid_traces,
                                       traces_gps=traces_gps,
                                       traces_lens=traces_lens,
                                       road_lens=road_lens,
                                       gdata=gdata,
                                       tf_ratio=tf_ratio)
            infer_seq = infer_seq.argmax(dim=-1).detach().cpu().numpy().flatten()
            tgt_roads = tgt_roads.flatten().numpy()
            mask = (tgt_roads != -1)
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


def main(args):
    save_path = "ckpt1/bz{}_lr{}_ep{}_locd{}_gcn{}_att{}_best_gclip.pt".format(
        args['batch_size'], args['lr'], args['epochs'], args['loc_dim'], args['use_gcn'], args['atten_flag'])
    root_path = "../data_for_GMM-Master/"
    trainset = MyDataset(root_path, "train")
    valset = MyDataset(root_path, "val")
    testset = MyDataset(root_path, "test")
    train_iter = DataLoader(dataset=trainset,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=padding)
    val_iter = DataLoader(dataset=valset,
                          batch_size=args['batch_size'],
                          collate_fn=padding)
    test_iter = DataLoader(dataset=testset,
                           batch_size=args['batch_size'],
                           collate_fn=padding)
    print("Loading Dataset Done!!!")
    # args['dev_id'] = 1 if args['use_gcn'] else 0
    device = torch.device(f"cuda:{args['dev_id']}" if torch.cuda.is_available() else "cpu")

    gdata = GraphData(parent_path=args['parent_path'],
                      layer=args['layer'],
                      device=device)
    print('get graph extra data finished!')
    model = GMM(loc_dim=args['loc_dim'],
                target_size=gdata.num_roads,
                beam_size=args['beam_size'],
                device=device,
                atten_flag=args['atten_flag'])
    model = model.to(device)
    # model.init_cache()
    best_acc = 0.
    print("Loading model Done!!!")
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.NLLLoss()
    optimizer = optim.AdamW(params=model.parameters(),
                            lr=args['lr'],
                            weight_decay=args['wd'])
    for e in range(args['epochs']):
        print(f"================Epoch: {e + 1}================")
        train_avg_loss = train(model, train_iter, loss_fn, optimizer, device, gdata, args['tf_ratio'])
        val_avg_acc, val_avg_r, val_avg_p = evaluate(model, val_iter, device, gdata, 0.)
        if best_acc <= val_avg_acc:
            best_acc = val_avg_acc
            torch.save(model.state_dict(), save_path)
        print("Epoch {}: train_avg_loss {} eval_avg_acc: {} eval_avg_r {} eval_avg_p {}"
            .format(e + 1, train_avg_loss, val_avg_acc, val_avg_r, val_avg_p))
        nni.report_intermediate_result(val_avg_acc)

    model.load_state_dict(torch.load(save_path))
    test_avg_acc, test_avg_r, test_avg_p = evaluate(model, test_iter, device, gdata, 0.)
    nni.report_final_result(test_avg_acc)
    print(f"testset: acc({test_avg_acc}) recall({test_avg_r}) precision({test_avg_p})")


if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter()
        if tuner_params and tuner_params['tf_ratio'] == 0:
            tuner_params['tf_ratio'] = 0.0
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        raise
