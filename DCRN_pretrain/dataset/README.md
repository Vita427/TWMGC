# Data directory

Every dataset is read from `DCRN_pretrain/dataset/<name>/`, with

```
<name>_feat.npy          node features, shape N x d
<name>_label.npy         ground-truth labels, shape N
<name>_adj.npy           adjacency, shape N x N (dense, 0/1)
<name>_adj_sparse.npz    optional, scipy sparse adjacency for large graphs
<name>_ppr_topk64.npz    optional, precomputed truncated PPR diffusion
```

The datasets used in the paper are:

| name | nodes | edges | features | classes | source |
|---|---|---|---|---|---|
| `acm` | 3025 | 13128 | 1870 | 3 | ACM citation network |
| `amap` | 7650 | 119081 | 745 | 8 | AMiner co-author network |
| `amac` | 13752 | 245861 | 767 | 10 | Amazon co-purchase network |
| `cocs` | 18333 | 81894 | 6805 | 15 | CoCS co-purchase network |
| `roman_empire` | 22662 | 32927 | 300 | 18 | Roman-empire (Platonov et al., ICLR 2023) |
| `amazon_ratings` | 24492 | 93050 | 300 | 5 | Amazon-ratings (Platonov et al., ICLR 2023) |

The two heterophilous graphs are distributed with PyTorch Geometric
(`torch_geometric.datasets.RomanEmpire`, `AmazonRatings`) and can be converted to the files above
with their public interfaces. Features are reduced to 100 dimensions by PCA inside
`utils.load_graph_data`, except for `roman_empire`, whose 300-dimensional features are used
directly.

The datasets themselves are not included in this repository because of their size and of the
licences under which they are distributed.
