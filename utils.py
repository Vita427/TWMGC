import os
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

import opt


# ----------------------------------------------------------------------------
# run setup
# ----------------------------------------------------------------------------
def setup():
    """Fill the per-dataset settings and the device.

    The loss weights of every dataset are the values reported in the paper.
    Any of them can be overridden from the command line.
    """
    print("setting:")
    setup_seed(opt.args.seed)

    # name: (n_clusters, n_input, beta, alpha, lr, l1, l2, l3)
    preset = {
        'acm':            (3,  100, 10,    10,    1e-3, 1e-3, 1e-2, 5e-1),
        'amap':           (8,  100, 1e-2,  1e-3,  1e-3, 1e-3, 1e-2, 5e-1),
        'amac':           (10, 100, 1e-2,  1e-4,  9e-4, 1e-1, 1e-2, 10),
        'cocs':           (15, 100, 1,     1e-3,  7e-4, 1e-2, 1e-2, 1e-1),
        'roman_empire':   (18, 300, 1,     1e-1,  1e-3, 1e-2, 1e-1, 10),
        'amazon_ratings': (5,  100, 1e-1,  2e-3,  2e-3, 1e-3, 1e-2, 5e-1),
    }
    if opt.args.name not in preset:
        raise SystemExit(
            "unknown dataset '{}'; add its settings to utils.setup()".format(opt.args.name))

    n_clusters, n_input, beta, alpha, lr, l1, l2, l3 = preset[opt.args.name]
    opt.args.n_clusters = n_clusters
    opt.args.n_input = n_input
    opt.args.lambda_value = beta if opt.args.lambda_value is None else opt.args.lambda_value
    opt.args.gamma_value = alpha if opt.args.gamma_value is None else opt.args.gamma_value
    opt.args.lr = lr if opt.args.lr is None else opt.args.lr
    opt.args.l1 = l1 if opt.args.l1 is None else opt.args.l1
    opt.args.l2 = l2 if opt.args.l2 is None else opt.args.l2
    opt.args.l3 = l3 if opt.args.l3 is None else opt.args.l3

    opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")

    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("random seed   : {}".format(opt.args.seed))
    print("clusters      : {}".format(opt.args.n_clusters))
    print("input dim     : {}".format(opt.args.n_input))
    print("eta (ppr)     : {}".format(opt.args.alpha_value))
    print("beta  (kl)    : {}".format(opt.args.lambda_value))
    print("alpha (rec)   : {}".format(opt.args.gamma_value))
    print("l1, l2, l3    : {}, {}, {}".format(opt.args.l1, opt.args.l2, opt.args.l3))
    print("learning rate : {}".format(opt.args.lr))
    print("------------------------------")


def setup_seed(seed):
    """Fix the random seed of every library used in the pipeline."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_graph_data(dataset_name, show_details=False):
    """Load the features, the labels and the adjacency of one dataset.

    The files are expected at ./DCRN_pretrain/dataset/<name>/<name>_{feat,label,adj}.npy,
    plus an optional <name>_adj_sparse.npz for large graphs.
    """
    load_path = "./DCRN_pretrain/dataset/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path + "_feat.npy", allow_pickle=True)
    label = np.load(load_path + "_label.npy", allow_pickle=True)

    # large graphs are stored as a scipy sparse matrix to avoid a dense N x N array
    if (dataset_name == "amazon_ratings" and opt.args.adj_format != "dense"
            and os.path.exists(load_path + "_adj_sparse.npz")):
        adj = sp.load_npz(load_path + "_adj_sparse.npz")
    else:
        adj = np.load(load_path + "_adj.npy", allow_pickle=True)

    if show_details:
        print("dataset: {}  features: {}  labels: {}".format(dataset_name, feat.shape, label.shape))
        print("classes: {}".format(int(label.max() - label.min() + 1)))

    # dimensionality reduction; the input dimension is fixed per dataset
    if opt.args.name in ["dblp"]:
        opt.args.n_input = feat.shape[1]
    else:
        feat = PCA(n_components=opt.args.n_input).fit_transform(feat)

    # the amazon features are already unit-variance in total, so after PCA each
    # dimension is around 0.06 and the tanh encoders collapse; rescale to std 1
    if opt.args.name == "amazon_ratings" and opt.args.adj_format != "dense":
        feat = feat / feat.std()

    return feat, label, adj


def normalize_adj(adj, self_loop=True, symmetry=False):
    """Return the normalised adjacency, keeping sparse input sparse.

    self_loop: add the identity matrix before normalising
    symmetry:  D^{-0.5} (A + I) D^{-0.5} instead of D^{-1} (A + I)
    """
    if sp.issparse(adj):
        adj_tmp = adj + sp.eye(adj.shape[0], format=adj.format) if self_loop else adj
        d_inv = 1.0 / np.asarray(adj_tmp.sum(1)).flatten()
        if symmetry:
            sqrt_d_inv = sp.diags(np.sqrt(d_inv))
            return (sqrt_d_inv @ adj_tmp @ sqrt_d_inv).tocsr()
        return (sp.diags(d_inv) @ adj_tmp).tocsr()

    adj_tmp = adj + np.eye(adj.shape[0]) if self_loop else adj
    d_inv = np.linalg.inv(np.diag(adj_tmp.sum(0)))
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        return np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)
    return np.matmul(d_inv, adj_tmp)


def gaussian_noised_feature(X):
    """Return the two feature matrices of the two views, each perturbed by its
    own Gaussian mask with mean 1 and standard deviation 0.1."""
    N_1 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(opt.args.device)
    N_2 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(opt.args.device)
    return X * N_1, X * N_2


def remove_edge(A, similarity, remove_rate=0.1):
    """Mask the edges with the lowest similarity and return the normalised
    adjacency of the resulting locally sparse view.

    For every row, round(remove_rate * N) entries with the lowest score are set
    to zero, so entries of the original adjacency are only removed and no new
    edge is introduced.
    """
    A = A.copy()
    n_node = A.shape[0]
    keep = int(round(remove_rate * n_node))
    for i in range(n_node):
        A[i, torch.argsort(similarity[i].cpu())[:keep]] = 0
    Am = normalize_adj(A, self_loop=True, symmetry=False)
    return numpy_to_torch(Am).to(opt.args.device)


# ----------------------------------------------------------------------------
# graph diffusion
# ----------------------------------------------------------------------------
def diffusion_adj(adj, mode="ppr", transport_rate=0.2):
    """Closed-form personalised PageRank diffusion for a dense adjacency.

    A^d = eta * (I - (1 - eta) * A_sym)^{-1}
    """
    adj_tmp = adj + np.eye(adj.shape[0])
    d_inv = np.linalg.inv(np.diag(adj_tmp.sum(0)))
    sqrt_d_inv = np.sqrt(d_inv)
    norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    if mode == "ppr":
        return transport_rate * np.linalg.inv(
            np.eye(norm_adj.shape[0]) - (1 - transport_rate) * norm_adj)
    return norm_adj


def _topk_rows(m, k):
    """Keep the k largest entries of every row of a scipy sparse matrix."""
    m = m.tocsr()
    indptr, indices, data = m.indptr, m.indices, m.data
    new_indices, new_data, new_indptr = [], [], [0]
    for i in range(m.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        row_data, row_idx = data[s:e], indices[s:e]
        if e - s > k:
            sel = np.argpartition(row_data, -k)[-k:]
            row_data, row_idx = row_data[sel], row_idx[sel]
        new_data.append(row_data)
        new_indices.append(row_idx)
        new_indptr.append(new_indptr[-1] + row_data.shape[0])
    return sp.csr_matrix(
        (np.concatenate(new_data), np.concatenate(new_indices), np.array(new_indptr)),
        shape=m.shape)


def diffusion_adj_sparse(adj, mode="ppr", transport_rate=0.2, topk=64, iters=10):
    """Sparse version of diffusion_adj for large graphs.

    The PPR matrix is expanded as a truncated power series and every row is
    pruned to its top-k entries after each step, so the matrix stays sparse.
    """
    n = adj.shape[0]
    adj_tmp = (adj + sp.eye(n, format='csr')).astype(np.float32)
    d_inv = 1.0 / np.asarray(adj_tmp.sum(1)).flatten()
    sqrt_d_inv = sp.diags(np.sqrt(d_inv))
    norm_adj = (sqrt_d_inv @ adj_tmp @ sqrt_d_inv).tocsr()

    if mode != "ppr":
        return norm_adj

    alpha = transport_rate
    diff_adj = alpha * sp.eye(n, format='csr', dtype=np.float32)
    power = sp.eye(n, format='csr', dtype=np.float32)
    coef = 1.0
    for _ in range(iters):
        coef *= (1.0 - alpha)
        power = _topk_rows(power @ norm_adj, topk)
        diff_adj = diff_adj + (alpha * coef) * power
    diff_adj = _topk_rows(diff_adj, topk)

    # the truncated series does not sum to one, so renormalise each row
    row_sum = np.asarray(diff_adj.sum(1)).flatten()
    row_sum[row_sum == 0] = 1.0
    return (sp.diags(1.0 / row_sum) @ diff_adj).tocsr()


# ----------------------------------------------------------------------------
# tensor helpers
# ----------------------------------------------------------------------------
def numpy_to_torch(a, sparse=False):
    if sparse:
        return torch.sparse.Tensor(a).to_sparse()
    return torch.FloatTensor(a)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ----------------------------------------------------------------------------
# pretraining and initialisation
# ----------------------------------------------------------------------------
def load_pretrain_parameter(model):
    """Load the pretrained AE/GAE weights into the model."""
    pretrain_path = 'DCRN_pretrain/pretrain/{}_pretrain.pkl'.format(opt.args.name)
    if not os.path.exists(pretrain_path):
        print("no pretrained model at {}, training from scratch".format(pretrain_path))
        return model
    pretrained_dict = torch.load(pretrain_path, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def model_init(model, X, y, A_norm):
    """Load the pretrained weights, then return the similarity matrix used for
    edge pruning and the initial cluster centres."""
    model = load_pretrain_parameter(model)
    with torch.no_grad():
        _, _, _, sim, _, _, _, _, _, Z = model(X, A_norm, X, A_norm)
    acc, nmi, ari, f1, centers, _ = clustering(Z, y)
    return sim, centers


# ----------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------
def reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat):
    """L_REC = L_AE + L_GAE, with L_GAE = L_w + rho * L_a (rho = 0.1)."""
    loss_ae = F.mse_loss(X_hat, X)
    loss_w = F.mse_loss(Z_hat, torch.spmm(A_norm, X))
    loss_a = F.mse_loss(A_hat, A_norm.to_dense())
    return loss_ae + loss_w + 0.1 * loss_a


def target_distribution(Q):
    """Target distribution S built from the soft assignment Q."""
    weight = Q ** 2 / Q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def distribution_loss(Q, P):
    """Average of the three KL divergences between the target S and the soft
    assignments of the pooled attribute, pooled structural and final
    representations, written as one KL between the geometric mean of the three
    assignments and the target."""
    return F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction='batchmean')


def CL_Loss(z1, z2, clusters, centers, label, temperature=0.5):
    """Contrastive loss of one representation space.

    z1, z2: the two views of the same representation
    centers, label: the k-means centres of this representation and the cluster
                    assigned to every node; the assigned centre is excluded and
                    the remaining c - 1 centres act as negatives
    """
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    centers = F.normalize(centers, p=2, dim=1)

    indicator = torch.ones(z1.size(0), clusters, device=z1.device)
    indicator[torch.arange(z1.size(0), device=z1.device), label.to(z1.device).long()] = 0

    sim_positive = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
    sim_negative = torch.exp(torch.mm(z1, centers.t()) / temperature) * indicator

    loss = -torch.log(sim_positive / (sim_positive + torch.sum(sim_negative, dim=1)))
    return loss.mean()


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def clustering(Z, y):
    """Cluster the embedding with k-means and return the four metrics."""
    model = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    cluster_id = model.fit_predict(Z.data.cpu().numpy())
    acc, nmi, ari, f1 = eva(y, cluster_id)
    return acc, nmi, ari, f1, model.cluster_centers_, cluster_id


def cluster_acc(y_true, y_pred):
    """ACC and macro F1 after matching cluster indices to class labels with the
    Hungarian algorithm."""
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            cost[i][j] = len([i1 for i1 in mps if y_pred[i1] == c2])
    m = Munkres()
    indexes = m.compute(cost.__neg__().tolist())
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred):
    """ACC, NMI, ARI and macro F1 of one clustering."""
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    return acc, nmi, ari, f1


# ----------------------------------------------------------------------------
# batching
# ----------------------------------------------------------------------------
def data_split(full_list, n_sample):
    """Shuffle the node indices and split them into batches."""
    random.shuffle(full_list)
    return [full_list[i:i + n_sample] for i in range(0, len(full_list), n_sample)]
