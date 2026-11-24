import csv
import datetime
import os
import resource
import time

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.cluster import KMeans
from torch.optim import Adam

import opt
from logger import Logger, metrics_info, record_info
from utils import (CL_Loss, clustering, data_split, distribution_loss,
                   gaussian_noised_feature, load_pretrain_parameter,
                   reconstruction_loss, remove_edge,
                   sparse_mx_to_torch_sparse_tensor, target_distribution)

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')


def _adj_batch(A_sp, batch):
    """Slice a sparse adjacency to one batch and move it to the device."""
    return sparse_mx_to_torch_sparse_tensor(
        A_sp[batch, :][:, batch]).to(opt.args.device)


def model_init_large(model, X, y, A_norm_sp, batch_size):
    """Initialisation for large graphs: the full-graph forward pass materialises
    N x N matrices, so it is run block by block."""
    model = load_pretrain_parameter(model)

    Z_list = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = list(range(start, min(start + batch_size, X.shape[0])))
            adj_batch = _adj_batch(A_norm_sp, batch)
            _, _, _, _, _, _, _, _, _, Z_batch = model(
                X[batch], adj_batch, X[batch], adj_batch)
            Z_list.append(Z_batch)
    Z = torch.cat(Z_list, dim=0)

    _, _, _, _, centers, _ = clustering(Z, y)
    return centers


def contrastive_losses(Z_ae_all, Z_gae_all):
    """Attribute-level, structural and global contrastive losses.

    For every representation an independent k-means provides the centres and the
    node assignments; the centre assigned to a node is excluded and the
    remaining c - 1 centres act as negatives.
    """
    def space_loss(z1, z2, km_anchor):
        km = KMeans(n_clusters=opt.args.n_clusters, n_init=1, init='k-means++')
        km.fit(km_anchor.data.cpu().numpy())
        centers = torch.FloatTensor(km.cluster_centers_).to(opt.args.device)
        label = torch.LongTensor(km.labels_).to(opt.args.device)
        forward = CL_Loss(z1, z2, opt.args.n_clusters, centers, label)
        backward = CL_Loss(z2, z1, opt.args.n_clusters, centers, label)
        return (forward + backward) / 2

    # attribute space: centres from the first and from the second AE view
    ae_cl = (space_loss(Z_ae_all[0], Z_ae_all[1], Z_ae_all[0]) +
             space_loss(Z_ae_all[0], Z_ae_all[1], Z_ae_all[1])) / 2
    # structural space: centres from the first and from the second GAE view
    gae_cl = (space_loss(Z_gae_all[0], Z_gae_all[1], Z_gae_all[0]) +
              space_loss(Z_gae_all[0], Z_gae_all[1], Z_gae_all[1])) / 2
    # global space: the two view-averaged representations
    Z_ae = (Z_ae_all[0] + Z_ae_all[1]) / 2
    Z_igae = (Z_gae_all[0] + Z_gae_all[1]) / 2
    global_cl = (space_loss(Z_ae, Z_igae, Z_ae) + space_loss(Z_ae, Z_igae, Z_igae)) / 2

    return ae_cl, gae_cl, global_cl


def train(model, X, y, A, A_norm, Ad, A_norm_sp=None, Ad_sp=None, setup_time=0.0):
    print("Training...")
    large = A_norm_sp is not None
    acc_result, nmi_result, ari_result, f1_result = [], [], [], []

    t_train_start = time.perf_counter()
    epoch_times = []
    n_params = sum(p.numel() for p in model.parameters())
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------- initialisation
    if large:
        centers = model_init_large(model, X, y, A_norm_sp, opt.args.batch_size)
        Am = None
    else:
        sim, centers = model_init(model, X, y, A_norm)
        # edge-masked adjacency of the locally sparse view
        Am = remove_edge(A, sim, remove_rate=0.1)

    model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)

    metrics = ['acc', 'nmi', 'ari', 'f1']
    logger = Logger(opt.args.name + '==' + nowtime)
    logger.info(model)
    logger.info(opt.args)
    logger.info(metrics_info(metrics))

    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.epoch)):
        _epoch_t0 = time.perf_counter()
        # Gaussian feature perturbation of the two views
        X_tilde1, X_tilde2 = gaussian_noised_feature(X)

        index = list(range(len(X_tilde1)))
        # full-graph training keeps several N x N tensors alive, so the large
        # datasets are trained in batches
        if opt.args.name in ["amac", "cocs", "corafull", "amazon_ratings", "roman_empire"]:
            split_list = data_split(index, opt.args.batch_size)
        else:
            split_list = [index]

        X_tilde1 = X_tilde1.to(opt.args.device)
        X_tilde2 = X_tilde2.to(opt.args.device)
        model = model.to(opt.args.device)

        for batch in split_list:
            X_tilde1_batch = X_tilde1[batch]
            X_tilde2_batch = X_tilde2[batch]

            if large:
                Ad_batch = _adj_batch(Ad_sp, batch)
                A_norm_batch = _adj_batch(A_norm_sp, batch)
                Am_batch = A_norm_batch
            else:
                Ad_batch = Ad[batch, :][:, batch].to(opt.args.device)
                Am_batch = Am[batch, :][:, batch].to(opt.args.device)

            X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, AZ_all, Z_all, Z = model(
                X_tilde1=X_tilde1_batch,
                Am=Am_batch,
                X_tilde2=X_tilde2_batch,
                Ad=Ad_batch
            )

            # reconstruction loss: L_AE + L_w + rho * L_a
            if large:
                L_REC = reconstruction_loss(X[batch], A_norm_batch, X_hat, Z_hat, A_hat)
            else:
                L_REC = reconstruction_loss(
                    X[batch], A_norm.to_dense()[batch, :][:, batch], X_hat, Z_hat, A_hat)

            # clustering loss toward the target distribution of the final view
            L_KL = distribution_loss(Q, target_distribution(Q[0].data))

            # three contrastive losses
            ae_CL, gae_CL, gae_global_CL = contrastive_losses(Z_ae_all, Z_gae_all)
            CL_Loss_total = (opt.args.l1 * ae_CL +
                             opt.args.l2 * gae_CL +
                             opt.args.l3 * gae_global_CL)

            # total loss
            loss = CL_Loss_total + opt.args.lambda_value * L_KL + opt.args.gamma_value * L_REC

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # ------------------------------------------------- evaluation
        if large:
            Z_list = []
            with torch.no_grad():
                for start in range(0, X.shape[0], opt.args.batch_size):
                    batch = list(range(start, min(start + opt.args.batch_size, X.shape[0])))
                    _, _, _, _, _, _, _, _, _, Z_batch = model(
                        X_tilde1=X_tilde1[batch],
                        Am=_adj_batch(A_norm_sp, batch),
                        X_tilde2=X_tilde2[batch],
                        Ad=_adj_batch(Ad_sp, batch)
                    )
                    Z_list.append(Z_batch)
            Z = torch.cat(Z_list, dim=0)
        else:
            with torch.no_grad():
                X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, AZ_all, Z_all, Z = model(
                    X_tilde1=X_tilde1, Am=Am, X_tilde2=X_tilde2, Ad=Ad)

        acc, nmi, ari, f1, _, pred = clustering(Z, y)
        acc_result.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)

        if use_cuda:
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - _epoch_t0)

        # write the representation and the labels once, at the last epoch
        if epoch != opt.args.epoch - 1:
            continue

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(Z.cpu()).to_excel(
            os.path.join(out_dir, '{}_fea.xlsx'.format(opt.args.name)), index=False)
        pd.DataFrame({'our': pred}).to_excel(
            os.path.join(out_dir, '{}_label.xlsx'.format(opt.args.name)), index=False)

    best = int(np.argmax(acc_result))
    best_acc, t_nmi, t_ari, t_f1 = acc_result[best], nmi_result[best], ari_result[best], f1_result[best]
    print("ACC: {:.4f}".format(best_acc))
    print("NMI: {:.4f}".format(t_nmi))
    print("ARI: {:.4f}".format(t_ari))
    print("F1: {:.4f}".format(t_f1))
    print("Epoch:", best)
    logger.info("best acc at epoch {}:\t{}".format(best, record_info([best_acc, t_nmi, t_ari, t_f1])))

    # ------------------------------------------------- time and memory
    if use_cuda:
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t_train_start
    total_time = setup_time + train_time
    epoch_avg, epoch_min, epoch_max = float(np.mean(epoch_times)), float(np.min(epoch_times)), float(np.max(epoch_times))
    if use_cuda:
        gpu_peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
        gpu_peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 20
    else:
        gpu_peak_alloc, gpu_peak_reserved = 0.0, 0.0
    cpu_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    n_nodes = int(X.shape[0])

    print("[Benchmark] name={} seed={} nodes={}".format(opt.args.name, opt.args.seed, n_nodes))
    print("[Benchmark] total_time={:.2f}s setup/preprocess={:.2f}s train={:.2f}s".format(
        total_time, setup_time, train_time))
    print("[Benchmark] epoch_avg={:.4f}s epoch_fastest={:.4f}s epoch_slowest={:.4f}s epochs={}".format(
        epoch_avg, epoch_min, epoch_max, len(epoch_times)))
    print("[Benchmark] gpu_peak_allocated={:.1f}MB gpu_peak_reserved={:.1f}MB "
          "cpu_peak_rss={:.1f}MB params={:.4f}M".format(
              gpu_peak_alloc, gpu_peak_reserved, cpu_peak_rss, n_params / 1e6))
    logger.info("[Benchmark] total_time={:.2f}s epoch_avg={:.4f}s epoch_fastest={:.4f}s "
                "gpu_peak_allocated={:.1f}MB gpu_peak_reserved={:.1f}MB cpu_peak_rss={:.1f}MB "
                "params={:.4f}M".format(total_time, epoch_avg, epoch_min, gpu_peak_alloc,
                                       gpu_peak_reserved, cpu_peak_rss, n_params / 1e6))

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results.csv')
    fields = ['name', 'seed', 'nodes', 'epoch_budget', 'best_epoch', 'acc', 'nmi', 'ari', 'f1',
              'total_time_s', 'setup_time_s', 'train_time_s', 'epoch_avg_s', 'epoch_min_s',
              'epoch_max_s', 'gpu_peak_alloc_mb', 'gpu_peak_reserved_mb', 'cpu_peak_rss_mb', 'params_m']
    row = [opt.args.name, opt.args.seed, n_nodes, opt.args.epoch, best,
           round(float(best_acc), 4), round(float(t_nmi), 4), round(float(t_ari), 4), round(float(t_f1), 4),
           round(total_time, 2), round(setup_time, 2), round(train_time, 2),
           round(epoch_avg, 4), round(epoch_min, 4), round(epoch_max, 4),
           round(gpu_peak_alloc, 1), round(gpu_peak_reserved, 1), round(cpu_peak_rss, 1),
           round(n_params / 1e6, 4)]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(fields)
        writer.writerow(row)
