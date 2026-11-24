import os

import scipy.sparse as sp

import opt
from DCRN import DCRN
from train import train
from utils import (diffusion_adj, diffusion_adj_sparse, load_graph_data,
                   normalize_adj, numpy_to_torch, setup,
                   sparse_mx_to_torch_sparse_tensor)
import time

if __name__ == '__main__':
    # ---------------------------------------------------------------- setup
    t_setup_start = time.perf_counter()
    setup()

    # ---------------------------------------------------------------- data
    X, y, A = load_graph_data(opt.args.name, show_details=False)

    # large graphs keep the adjacency sparse end to end; the other datasets use
    # the dense pipeline
    if sp.issparse(A):
        A_norm_sp = normalize_adj(A, self_loop=True, symmetry=False)

        # precomputed truncated PPR; the dense inverse is infeasible at this scale
        ppr_path = os.path.join('DCRN_pretrain', 'dataset', opt.args.name,
                                opt.args.name + '_ppr_topk64.npz')
        if os.path.exists(ppr_path):
            print("loading precomputed PPR diffusion from {}".format(ppr_path))
            Ad_sp = sp.load_npz(ppr_path)
        else:
            print("computing truncated PPR diffusion for {} nodes".format(A.shape[0]))
            Ad_sp = diffusion_adj_sparse(A, mode="ppr",
                                         transport_rate=opt.args.alpha_value,
                                         topk=64, iters=10)

        X = numpy_to_torch(X).to(opt.args.device)
        A_norm = sparse_mx_to_torch_sparse_tensor(A_norm_sp).to(opt.args.device)
        Ad = sparse_mx_to_torch_sparse_tensor(Ad_sp).to(opt.args.device)
    else:
        A_norm_sp, Ad_sp = None, None
        A_norm = normalize_adj(A, self_loop=True, symmetry=False)
        Ad = diffusion_adj(A, mode="ppr", transport_rate=opt.args.alpha_value)

        X = numpy_to_torch(X).to(opt.args.device)
        A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
        Ad = numpy_to_torch(Ad).to(opt.args.device)

    # ---------------------------------------------------------------- model
    model = DCRN(n_node=X.shape[0]).to(opt.args.device)
    print(model)

    # ---------------------------------------------------------------- training
    setup_time = time.perf_counter() - t_setup_start
    train(model, X, y, A, A_norm, Ad, A_norm_sp=A_norm_sp, Ad_sp=Ad_sp,
          setup_time=setup_time)
