import argparse

parser = argparse.ArgumentParser(
    description='TWMGC: triple-contrast learning and weighted cross-view fusion for multi-view graph clustering',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# ----------------------------- run setting -----------------------------
parser.add_argument('--name', type=str, default='acm',
                    help='dataset name: acm, amap, amac, cocs, roman_empire or amazon_ratings')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=5000)
# adjacency format: 'auto' keeps the sparse pipeline for large graphs,
# 'dense' forces the dense pipeline used by the moderate-size datasets
parser.add_argument('--adj_format', type=str, default='auto',
                    choices=['auto', 'sparse', 'dense'])

# ----------------------------- losses -----------------------------
# eta, restart probability of the PPR diffusion used to build the second view
parser.add_argument('--alpha_value', type=float, default=0.2)
# beta, weight of the clustering (KL) term
parser.add_argument('--lambda_value', type=float, default=None,
                    help='overrides the per-dataset value when given')
# alpha, weight of the reconstruction term
parser.add_argument('--gamma_value', type=float, default=None,
                    help='overrides the per-dataset value when given')
# l1, l2, l3, weights of the attribute, structural and global contrastive terms
parser.add_argument('--l1', type=float, default=None,
                    help='overrides the per-dataset value when given')
parser.add_argument('--l2', type=float, default=None,
                    help='overrides the per-dataset value when given')
parser.add_argument('--l3', type=float, default=None,
                    help='overrides the per-dataset value when given')

# ----------------------------- optimisation -----------------------------
parser.add_argument('--lr', type=float, default=None,
                    help='overrides the per-dataset value when given')
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--show_training_details', type=bool, default=False)

# ----------------------------- AE / GAE widths -----------------------------
parser.add_argument('--ae_n_enc_1', type=int, default=128)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=512)
parser.add_argument('--ae_n_dec_1', type=int, default=512)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=128)

parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=256)
parser.add_argument('--gae_n_enc_3', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)

args = parser.parse_args()
