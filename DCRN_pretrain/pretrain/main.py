import opt
import torch
import random
from AE import AE
import numpy as np
import scipy.sparse as sp
from opt import args
from GAE import IGAE
from torch import nn
from utils import eva
from torch.optim import Adam
from torch.nn import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from sklearn.decomposition import PCA

if args.name == 'acm':
    args.n_clusters = 3
    args.n_input = 100
    args.lr = 1e-3
    args.alpha = 1e-1
    args.beta = 1e-2
    args.omega = 1e-1

elif args.name == 'cite':
    args.n_clusters = 6
    args.n_input = 100
    args.lr = 2e-3 #1e-4 |1e-3
    args.alpha = 1e-1
    args.beta = 1e-2
    args.omega = 1e-1

elif args.name == 'cora':
    args.n_clusters = 7
    args.n_input = 100#100
    args.lr = 1e-3

elif args.name == 'amap':
    args.n_clusters = 8
    args.n_input = 100
    args.lr = 1e-3

elif args.name == 'amac':
    args.n_clusters = 10
    args.n_input = 100
    args.lr = 1e-3 #5e-4

elif args.name == 'cocs':
    args.n_clusters = 15
    args.n_input = 100
    args.lr = 1e-4

elif args.name == 'dblp':
    args.n_clusters = 4
    args.n_input = 100
    args.lr = 4e-3
    args.alpha = 1e-1
    args.beta = 1e-2
    args.omega = 1e-1

elif args.name == 'pubmed':
    args.n_clusters = 3
    args.n_input = 300
    args.lr =1e-5 #5e-3

elif args.name == 'corafull':
    args.n_clusters = 70
    args.n_input = 300
    args.lr =1e-5 #5e-3

elif args.name == 'uat':
    args.n_clusters = 4
    args.n_input = 239
    args.lr = 1e-4 #1e-4

elif args.name == 'flickr':
    args.n_clusters = 7
    args.n_input = 100
    args.lr = 7e-4

elif args.name == 'amazon_ratings':
    args.n_clusters = 5
    args.n_input = 100
    args.lr = 1e-3

elif args.name == 'roman_empire':
    args.n_clusters = 18
    args.n_input = 300
    args.lr = 1e-3

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

def data_split(full_list, n_sample):
    offset = n_sample
    random.shuffle(full_list)
    len_all = len(full_list)
    index_now = 0
    split_list = []
    while index_now < len_all:
        # 0-2000
        if index_now+offset > len_all:
            split_list.append(full_list[index_now:len_all])
        else:
            split_list.append(full_list[index_now:index_now+offset])
        index_now += offset
    return split_list

def load_graph_data(dataset_name, show_details=False):
    print(dataset_name)
    load_path = "../dataset/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)

    # Load adjacency matrix - use sparse format for large graphs
    if dataset_name in ("flickr", "amazon_ratings"):
        sparse_path = load_path + "_adj_sparse.npz"
        import os
        if os.path.exists(sparse_path):
            adj = sp.load_npz(sparse_path)
        else:
            adj = np.load(load_path+"_adj.npy", allow_pickle=True)
            adj = sp.csr_matrix(adj)
    else:
        adj = np.load(load_path+"_adj.npy", allow_pickle=True)

    return feat, label, adj

class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix (numpy or scipy sparse)
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix (same type as input)
    """
    is_sparse = sp.issparse(adj)

    if is_sparse:
        if self_loop:
            adj_tmp = adj + sp.eye(adj.shape[0], format=adj.format)
        else:
            adj_tmp = adj

        d = np.array(adj_tmp.sum(1)).flatten()
        d_inv = 1.0 / d

        if symmetry:
            sqrt_d_inv = np.sqrt(d_inv)
            sqrt_d_inv_mat = sp.diags(sqrt_d_inv)
            norm_adj = (sqrt_d_inv_mat @ adj_tmp @ sqrt_d_inv_mat).tocsr()
        else:
            d_inv_mat = sp.diags(d_inv)
            norm_adj = (d_inv_mat @ adj_tmp).tocsr()

        return norm_adj, adj_tmp
    else:
        if self_loop:
            adj_tmp = adj + np.eye(adj.shape[0])
        else:
            adj_tmp = adj

        d = np.diag(adj_tmp.sum(0))
        d_inv = np.linalg.inv(d)

        if symmetry:
            sqrt_d_inv = np.sqrt(d_inv)
            norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)
        else:
            norm_adj = np.matmul(d_inv, adj_tmp)

        return norm_adj, adj_tmp

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class Pre_model(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_input, n_z, n_clusters, v=1.0, n_node=None, device=None):
        super(Pre_model, self).__init__()

        self.ae = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

        self.ae.load_state_dict(torch.load(args.ae_model_save_path))

        self.gae = IGAE(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)

        self.gae.load_state_dict(torch.load(args.gae_model_save_path))

        self.a = Parameter(nn.init.constant_(torch.zeros(1), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(1), 0.5), requires_grad=True)

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj):
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.gae.encoder(x, adj)
        z_i = self.a * z_ae + self.b * z_igae
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.gamma * z_g + z_l
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj

        return x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde

print("network setting...")
print("use cuda: {}".format(opt.args.cuda))
device = torch.device("cuda" if opt.args.cuda else "cpu")

opt.args.ae_model_save_path = '../pretrain_ae/{}_ae.pkl'.format(opt.args.name)
opt.args.gae_model_save_path = '../pretrain_gae/{}_gae.pkl'.format(opt.args.name)

opt.args.pre_model_save_path = './{}_pretrain.pkl'.format(opt.args.name)

x, y, adj = load_graph_data(opt.args.name)
is_sparse_adj = sp.issparse(adj)
norm_adj, adj_tmp = normalize_adj(adj, self_loop=True, symmetry=False)

pca = PCA(n_components=opt.args.n_input)
X_pca = pca.fit_transform(x)
if args.name == "amazon_ratings":
    X_pca = X_pca / X_pca.std()

if args.name in ["dblp","bat","eat", "uat"]:
    dataset = LoadDataset(x)
    data = torch.Tensor(dataset.x).to(device)
    args.n_input = data.shape[1]
else:
    dataset = LoadDataset(X_pca)
    data = torch.Tensor(dataset.x).to(device)

if is_sparse_adj:
    if norm_adj.shape[0] > 10000:
        adj = norm_adj
    else:
        adj = torch.FloatTensor(norm_adj.toarray()).to(device)
else:
    if norm_adj.shape[0] > 10000:
        adj = norm_adj
    else:
        adj = torch.FloatTensor(norm_adj).to(device)
label = y

model = Pre_model(ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2, ae_n_enc_3=opt.args.ae_n_enc_3,
                  ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2, ae_n_dec_3=opt.args.ae_n_dec_3,
                  gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2, gae_n_enc_3=opt.args.gae_n_enc_3,
                  gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2, gae_n_dec_3=opt.args.gae_n_dec_3,
                  n_input=opt.args.n_input,
                  n_z=opt.args.n_z,
                  n_clusters=opt.args.n_clusters,
                  v=opt.args.freedom_degree,
                  n_node=data.size()[0],
                  device=device).to(device)

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
def Pretrain(model, data, adj, label):
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        index = list(range(len(data)))
        if args.name in ["amac", "cocs", "corafull", "flickr", "amazon_ratings", "roman_empire"]:
            split_list = data_split(index, args.batch_size)
        else:
            split_list = [index]
        for batch in split_list:
            data_batch = data[batch]

            # Handle both torch tensors and numpy/scipy matrices
            if isinstance(adj, torch.Tensor):
                adj_batch = adj[batch, :][:, batch]
            else:
                adj_batch = adj[batch, :][:, batch]
                if sp.issparse(adj_batch):
                    adj_batch = torch.FloatTensor(adj_batch.toarray()).to(device)
                else:
                    adj_batch = torch.FloatTensor(adj_batch).to(device)

            x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde = model(data_batch, adj_batch)

            loss_1 = F.mse_loss(x_hat, data_batch)
            loss_2 = F.mse_loss(z_hat, torch.spmm(adj_batch, data_batch))
            loss_3 = F.mse_loss(adj_hat, adj_batch)

            loss_4 = F.mse_loss(z_ae, z_igae)
            loss = loss_1 + args.alpha * loss_2 + args.beta * loss_3 + args.omega * loss_4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # For large graphs, process in batches to avoid OOM
            if isinstance(adj, torch.Tensor) and adj.shape[0] <= 10000:
                x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde = model(data, adj)
            else:
                # Process in batches
                z_tilde_list = []
                batch_size_eval = 5000
                n_nodes = data.shape[0]
                for i in range(0, n_nodes, batch_size_eval):
                    batch_idx = list(range(i, min(i + batch_size_eval, n_nodes)))
                    data_batch = data[batch_idx]

                    if isinstance(adj, torch.Tensor):
                        adj_batch = adj[batch_idx, :][:, batch_idx]
                    else:
                        adj_batch_slice = adj[batch_idx, :][:, batch_idx]
                        if sp.issparse(adj_batch_slice):
                            adj_batch = torch.FloatTensor(adj_batch_slice.toarray()).to(device)
                        else:
                            adj_batch = torch.FloatTensor(adj_batch_slice).to(device)

                    _, _, _, _, _, z_tilde_batch = model(data_batch, adj_batch)
                    z_tilde_list.append(z_tilde_batch.cpu())

                z_tilde = torch.cat(z_tilde_list, dim=0).to(device)

        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())
        acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)
        torch.save(model.state_dict(), args.pre_model_save_path)

    print("Finish optimization.")
    print("ACC: {:.4f}".format(max(acc_reuslt)))
    print("NMI: {:.4f}".format(nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("ARI: {:.4f}".format(ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("F1: {:.4f}".format(f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("Epoch:", np.where(acc_reuslt == np.max(acc_reuslt))[0][0])

Pretrain(model, data, adj, label)

