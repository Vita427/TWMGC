import torch
import random
from AE import AE
from opt import args
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from utils import eva
from sklearn.cluster import KMeans

if args.name == 'acm':
    args.n_clusters = 3
    args.n_input = 100
    args.lr = 1e-3
    args.s = 10

elif args.name == 'cite':
    args.n_clusters = 6
    args.n_input = 100
    args.lr = 1e-3
    args.s = 1#0

elif args.name == 'cora':
    args.n_clusters = 7
    args.n_input = 100
    args.lr = 2e-3
    args.s = 10

elif args.name == 'amap':
    args.n_clusters = 8
    args.n_input = 100
    args.lr = 1e-3
    args.s = 1#0

elif args.name == 'amac':
    args.n_clusters = 10
    args.n_input = 100 #767
    args.lr = 8e-4 #8e-4 #yuanlaishi7e-4
    args.s = 10

elif args.name == 'cocs':
    args.n_clusters = 15
    args.n_input = 100
    args.lr = 1e-3
    args.s = 10

elif args.name == 'dblp':
    args.n_clusters = 4
    args.n_input = 50
    args.lr = 4e-3
    args.s = 10

elif args.name == 'uat':
    args.n_clusters = 4
    args.n_input = 239
    args.lr = 1e-3
    args.s = 10

elif args.name == 'eat':
    args.n_clusters = 4
    args.n_input = 203
    args.lr = 4e-3
    args.s = 10

elif args.name == 'film':
    args.n_clusters = 5
    args.n_input = 100
    args.lr = 1e-3
    args.s = 10

elif args.name == 'pubmed':
    args.n_clusters = 3
    args.n_input = 300
    args.lr =1e-4 #1e-5

elif args.name == 'corafull':
    args.n_clusters = 70
    args.n_input = 300
    args.lr =1e-3 #1e-3

elif args.name == 'flickr':
    args.n_clusters = 7
    args.n_input = 100
    args.lr = 1e-3
    args.s = 10

elif args.name == 'amazon_ratings':
    args.n_clusters = 5
    args.n_input = 100
    args.lr = 1e-3
    args.s = 10

elif args.name == 'roman_empire':
    args.n_clusters = 18
    args.n_input = 300
    args.lr = 1e-3
    args.s = 10

else:
    print("error!")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(1)
print("use cuda: {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
   device = torch.device("cuda")
else:
   device = torch.device("cpu")
args.device = device

def load_graph_data(dataset_name):
    print(dataset_name)
    load_path = "../dataset/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    
    # AE pretraining doesn't use adjacency matrix, so we can skip loading it
    # Just return a dummy value
    adj = None
    
    return feat, label, adj

args.model_save_path = '{}_ae.pkl'.format(args.name)

class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []

def Pretrain_ae(model,dataset,y, train_loader, device):
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_hat, z_hat = model(x)
            loss = F.mse_loss(x_hat, x)*args.s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), args.model_save_path)

        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).float()
            x_hat,z_hat = model(x)

        print('{} loss: {}'.format(epoch, loss))
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z_hat.data.cpu().numpy())
        pred = kmeans.fit_predict(z_hat.data.cpu().numpy())
        acc, nmi, ari, f1= eva(y, pred)
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)
        
    print("Finish optimization.")
    
    print("ACC: {:.4f}".format(max(acc_reuslt)))
    print("NMI: {:.4f}".format(nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("ARI: {:.4f}".format(ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("F1: {:.4f}".format(f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("Epoch:", np.where(acc_reuslt == np.max(acc_reuslt))[0][0])

x, y, adj = load_graph_data(args.name)

pca = PCA(n_components=args.n_input)
X_pca = pca.fit_transform(x)
if args.name == "amazon_ratings":
    X_pca = X_pca / X_pca.std()

if args.name in ["dblp","uat","eat"]:
    dataset = LoadDataset(x)
    data = torch.Tensor(dataset.x).to(args.device)
    args.n_input = data.shape[1]
else:
    dataset = LoadDataset(X_pca)

if args.name in ["acm","cite","dblp","uat","amac","film"]: #
    args.batch_size = 256
else:
    args.batch_size = 5000

train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

model = AE(
    ae_n_enc_1=args.ae_n_enc_1,
    ae_n_enc_2=args.ae_n_enc_2,
    ae_n_enc_3=args.ae_n_enc_3,
    ae_n_dec_1=args.ae_n_dec_1,
    ae_n_dec_2=args.ae_n_dec_2,
    ae_n_dec_3=args.ae_n_dec_3,
    n_input=args.n_input,
    n_z=args.n_z).to(device)

Pretrain_ae(model,dataset,y, train_loader, device)
