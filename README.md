# TWMGC

Reference implementation of **Multi-view Graph Clustering Based on Triple-Contrast Learning and
Weighted Cross-View Feature Fusion (TWMGC)**.

Two views of the same attributed graph are built by Gaussian feature perturbation combined with
two different graph transformations:

* a **locally sparse view**: edges with the lowest similarity scores are pruned, and the masked
  adjacency is one-side normalised;
* a **globally smoothed view**: a personalised-PageRank diffusion matrix, computed either in closed
  form or, on the large graphs, as a truncated power series with top-64 row pruning.

An AE and a GAE with shared parameters across the two views learn attribute (`H`) and structural
(`Z`) representations. Three contrastive objectives act on the attribute space, the structural space
and the two view-averaged representations, using the cluster centroids not assigned to a node as
negative prototypes. A dual-branch fusion module combines the four view-specific embeddings through
learnable node-wise weights (MLFA) and the two pooled representations through a shared attention
network propagated over the sparse graph (GFA); the two branches are then combined by learnable
scalar coefficients. A self-supervised objective guides the pooled attribute, pooled structural and
final representations toward one shared target distribution.

## Repository layout

```
TWMGC/
├── main.py                     entry point: setup, data, diffusion, model, training
├── train.py                    training loop, evaluation, timing and memory report
├── DCRN.py                     AE / GAE encoders and decoders, attention, soft assignment
├── MSC.py                      MLFA: node-wise weighting of the four view-specific embeddings
├── GCN.py                      graph propagation layer used by the GAE
├── utils.py                    data loading, normalisation, diffusion, losses, metrics
├── logger.py                   console + file logger
├── opt.py                      command-line arguments
├── run_all.sh                  runs every dataset with ten seeds
└── DCRN_pretrain/
    ├── dataset/                data go here (see below)
    ├── pretrain_ae/            AE pretraining            → <dataset>_ae.pkl
    ├── pretrain_gae/           GAE pretraining           → <dataset>_gae.pkl
    └── pretrain/              joint DFCN-style pretraining → <dataset>_pretrain.pkl
```

The three `*.pkl` files of the six datasets of the paper are already in the tree, so training can
start without rerunning the pretraining.

## Environment

```
python >= 3.8
pip install -r requirements.txt
```

The experiments of the paper were run on a single NVIDIA RTX 4090 with 24 GB of memory, PyTorch,
and a fixed seed for every run.

## Data

Place every dataset under `DCRN_pretrain/dataset/<name>/` with these files:

| file | content |
|---|---|
| `<name>_feat.npy` | node features, shape `N x d` |
| `<name>_label.npy` | ground-truth labels, shape `N` |
| `<name>_adj.npy` | adjacency, shape `N x N` (dense) |
| `<name>_adj_sparse.npz` | *optional*, scipy sparse adjacency for large graphs |
| `<name>_ppr_topk64.npz` | *optional*, precomputed truncated PPR diffusion |

If the last two files are absent, the sparse adjacency is read from `<name>_adj.npy` and the
truncated PPR diffusion is computed on the fly. Features are reduced to 100 dimensions by PCA
(300 for `roman_empire`, whose features are used directly); for `amazon_ratings` the reduced
features are rescaled to unit standard deviation.

The six datasets of the paper are ACM, AMAP, AMAC, COCS, Roman-empire and Amazon-ratings; their
preprocessing details are listed in Table 1 of the paper.

## Pretraining

Run from inside the corresponding directory (each script writes its `*.pkl` next to itself):

```
cd DCRN_pretrain/pretrain_ae  && python main.py --name acm
cd DCRN_pretrain/pretrain_gae && python main.py --name acm
cd DCRN_pretrain/pretrain     && python main.py --name acm --epoch 100
```

`--epoch 100` matches the 100 pretraining iterations of Section 3.2 of the paper.

## Training

```
python main.py --name acm --seed 0
```

The per-dataset settings of Section 3.2 of the paper are stored in `utils.setup()` and are used by
default; every weight and the learning rate can be overridden from the command line, for example

```
python main.py --name cocs --seed 0 --l1 1e-2 --l2 1e-2 --l3 1e-1 --lambda_value 1 --gamma_value 1e-3 --lr 7e-4
```

Reported defaults:

| dataset | k | dim | l1 | l2 | l3 | beta | alpha | lr |
|---|---|---|---|---|---|---|---|---|
| acm | 3 | 100 | 1e-3 | 1e-2 | 5e-1 | 10 | 10 | 1e-3 |
| amap | 8 | 100 | 1e-3 | 1e-2 | 5e-1 | 1e-2 | 1e-3 | 1e-3 |
| amac | 10 | 100 | 1e-1 | 1e-2 | 10 | 1e-2 | 1e-4 | 9e-4 |
| cocs | 15 | 100 | 1e-2 | 1e-2 | 1e-1 | 1 | 1e-3 | 7e-4 |
| roman_empire | 18 | 300 | 1e-2 | 1e-1 | 10 | 1 | 1e-1 | 1e-3 |
| amazon_ratings | 5 | 100 | 1e-3 | 1e-2 | 5e-1 | 1e-1 | 2e-3 | 2e-3 |

`l1`, `l2`, `l3` are the weights of the attribute, structural and global contrastive losses,
`beta` the weight of the clustering term and `alpha` the weight of the reconstruction term; the
PPR restart probability is `eta = 0.2` for every dataset.

`python main.py --help` lists all arguments, including the network widths
(AE `d-128-256-512-20`, GAE `d-128-256-20`, attention hidden layer of 64 units).

## Output

For every run the script prints ACC, NMI, ARI and F1 at the best epoch, writes the log to
`log/<dataset>==<timestamp>.log`, appends one row per run to `benchmark_results.csv`
(time, peak GPU memory, parameter count) and saves the final representation and the predicted
labels to `output/<dataset>_{fea,label}.xlsx`.

## Notes

* Mini-batch training is used for `amac`, `cocs`, `roman_empire` and `amazon_ratings`; the other
  datasets are trained on the full graph. The batch size is set by `--batch_size`.
* The memory is dominated by the `N x N` adjacency and diffusion matrices, and by the adjacency
  reconstruction inside every epoch; large graphs therefore need a GPU with enough memory, and
  `--adj_format dense` should only be used for the moderate-size datasets.
* The prototype of every contrastive term is recomputed by k-means++ at each step, and the cluster
  centres of the self-supervised term are initialised once by k-means and then updated jointly with
  the network parameters.
* `run_all.sh` reproduces the setting of the paper: ten seeds per dataset.

## Citation

```
@article{twmgc,
  title   = {Multi-view Graph Clustering Based on Triple-Contrast Learning and Weighted Cross-View Feature Fusion},
  journal = {Information Processing \& Management},
  year    = {2026}
}
```
