from model.gmm import GMM
from config import get_params
from graph_data_gps import GraphData
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


args = get_params()
args = vars(args)


device = torch.device(f"cuda:{args['dev_id']}" if torch.cuda.is_available() else "cpu")

gdata = GraphData(parent_path=args['parent_path'],
                    layer=args['layer'],
                    device=device)

model = GMM(loc_dim=args['loc_dim'],
            target_size=gdata.num_roads,
            beam_size=args['beam_size'],
            device=device,
            atten_flag=args['atten_flag'])
save_path = "ckpt/bz256_lr0.0005_ep200_locd32_gcn1_att1_best_gclip.pt"
model.load_state_dict(torch.load(save_path))
model = model.to(device)

full_road_emb = model(None, None, None, None, None, gdata, 0.5).detach().cpu().numpy()
pca = PCA(n_components=2)
pca.fit(full_road_emb)
X_new = pca.transform(full_road_emb)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')
plt.savefig("images/gcn_org.png")