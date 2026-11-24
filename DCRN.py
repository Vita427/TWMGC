import torch
from torch import nn
from torch.nn import Linear, Module, Parameter

import opt
from GCN import GNNLayer
from MSC import MSC


class Attention(nn.Module):
    """Node-wise attention over the two view-averaged representations.

    The two representations are stacked, scored by a shared two-layer network
    and normalised across the two scores, as in Eq. (34)-(38) of the paper.
    """

    def __init__(self, emb_dim, hidden_size=64):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            Linear(emb_dim, hidden_size),
            nn.ReLU(),
            Linear(hidden_size, 1, bias=False),
        )

    def forward(self, H1, H2):
        H = torch.stack([H1, H2], dim=1)
        w = self.project(H)
        beta = torch.softmax(w, dim=1)
        return (beta * H).sum(1)


class AE_encoder(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)
        self.z_layer = Linear(ae_n_enc_3, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z = self.act(self.enc_3(z))
        return self.z_layer(z)


class AE_decoder(nn.Module):
    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE_decoder, self).__init__()
        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        z = self.act(self.dec_3(z))
        return self.x_bar_layer(z)


class AE(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE, self).__init__()
        self.encoder = AE_encoder(ae_n_enc_1=ae_n_enc_1,
                                  ae_n_enc_2=ae_n_enc_2,
                                  ae_n_enc_3=ae_n_enc_3,
                                  n_input=n_input,
                                  n_z=n_z)
        self.decoder = AE_decoder(ae_n_dec_1=ae_n_dec_1,
                                  ae_n_dec_2=ae_n_dec_2,
                                  ae_n_dec_3=ae_n_dec_3,
                                  n_input=n_input,
                                  n_z=n_z)


class IGAE_encoder(nn.Module):
    """Graph encoder: three propagation layers with tanh activation."""

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z_1, az_1 = self.gnn_1(x, adj, active=True)
        z_2, az_2 = self.gnn_2(z_1, adj, active=True)
        z_igae, az_3 = self.gnn_3(z_2, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj, [az_1, az_2, az_3], [z_1, z_2, z_igae]


class IGAE_decoder(nn.Module):
    """Graph decoder: three propagation layers back to the input dimension."""

    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z_1, az_1 = self.gnn_4(z_igae, adj, active=True)
        z_2, az_2 = self.gnn_5(z_1, adj, active=True)
        z_hat, az_3 = self.gnn_6(z_2, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj, [az_1, az_2, az_3], [z_1, z_2, z_hat]


class IGAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(gae_n_enc_1=gae_n_enc_1,
                                    gae_n_enc_2=gae_n_enc_2,
                                    gae_n_enc_3=gae_n_enc_3,
                                    n_input=n_input)
        self.decoder = IGAE_decoder(gae_n_dec_1=gae_n_dec_1,
                                    gae_n_dec_2=gae_n_dec_2,
                                    gae_n_dec_3=gae_n_dec_3,
                                    n_input=n_input)


class DCRN(nn.Module):
    """Two-view clustering network: shared AE and GAE encoders, the MLFA module
    (MSC), the GFA module (Attention) and the soft assignment layer."""

    def __init__(self, n_node=None):
        super(DCRN, self).__init__()

        self.ae = AE(ae_n_enc_1=opt.args.ae_n_enc_1,
                     ae_n_enc_2=opt.args.ae_n_enc_2,
                     ae_n_enc_3=opt.args.ae_n_enc_3,
                     ae_n_dec_1=opt.args.ae_n_dec_1,
                     ae_n_dec_2=opt.args.ae_n_dec_2,
                     ae_n_dec_3=opt.args.ae_n_dec_3,
                     n_input=opt.args.n_input,
                     n_z=opt.args.n_z)

        self.gae = IGAE(gae_n_enc_1=opt.args.gae_n_enc_1,
                        gae_n_enc_2=opt.args.gae_n_enc_2,
                        gae_n_enc_3=opt.args.gae_n_enc_3,
                        gae_n_dec_1=opt.args.gae_n_dec_1,
                        gae_n_dec_2=opt.args.gae_n_dec_2,
                        gae_n_dec_3=opt.args.gae_n_dec_3,
                        n_input=opt.args.n_input)

        # learnable coefficients of the weighted cross-view fusion, Eq. (40)
        self.a = Parameter(nn.init.constant_(torch.zeros(1), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(1), 0.5), requires_grad=True)

        # learnable cluster centres shared by the three soft assignments
        self.cluster_centers = Parameter(
            torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)

        # MLFA and GFA
        self.msc = MSC(opt.args.n_z)
        self.Attention = Attention(opt.args.gae_n_enc_3)

    def q_distribute(self, Z, Z_ae, Z_igae):
        """Soft assignments of the final, pooled attribute and pooled structural
        representations against the shared cluster centres."""
        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_centers, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(Z_ae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_igae = 1.0 / (1.0 + torch.sum(torch.pow(Z_igae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_igae = (q_igae.t() / torch.sum(q_igae, 1)).t()

        return [q, q_ae, q_igae]

    def forward(self, X_tilde1, Am, X_tilde2, Ad):
        # view 1 uses the locally sparse graph Am, view 2 the diffused graph Ad
        Z_ae1 = self.ae.encoder(X_tilde1)
        Z_ae2 = self.ae.encoder(X_tilde2)

        Z_igae1, A_igae1, AZ_1, Z_1 = self.gae.encoder(X_tilde1, Am)
        Z_igae2, A_igae2, AZ_2, Z_2 = self.gae.encoder(X_tilde2, Ad)

        Z_ae = (Z_ae1 + Z_ae2) / 2
        Z_igae = (Z_igae1 + Z_igae2) / 2

        # MLFA: node-wise weighting of the four view-specific embeddings
        Z = self.msc(Z_ae1, Z_ae2, Z_igae1, Z_igae2, Am)
        # GFA: attention over the two pooled representations, propagated over Am
        Z_i = self.Attention(Z_ae, Z_igae)
        Z_l = torch.spmm(Am, Z_i)
        # weighted cross-view fusion, Eq. (40)
        Z = self.a * Z + self.b * Z_l

        # the fused representation is decoded by both decoders
        X_hat = self.ae.decoder(Z)
        Z_hat, Z_adj_hat, AZ_de, Z_de = self.gae.decoder(Z, Am)
        A_hat = Z_adj_hat

        # encoder-side similarity, used to prune edges at initialisation
        sim = (A_igae1 + A_igae2) / 2

        Z_ae_all = [Z_ae1, Z_ae2]
        Z_gae_all = [Z_igae1, Z_igae2]
        Q = self.q_distribute(Z, Z_ae, Z_igae)

        AZ_en, Z_en = [], []
        for i in range(len(AZ_1)):
            AZ_en.append((AZ_1[i] + AZ_2[i]) / 2)
            Z_en.append((Z_1[i] + Z_2[i]) / 2)

        AZ_all = [AZ_en, AZ_de]
        Z_all = [Z_en, Z_de]

        return X_hat, Z_hat, A_hat, sim, Z_ae_all, Z_gae_all, Q, AZ_all, Z_all, Z
