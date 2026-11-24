import torch
import torch.nn.functional as F
from torch.nn import Linear, Module

import opt


class MSC(Module):
    """Multi-level feature alignment (MLFA).

    The four view-specific embeddings are concatenated, scored by a learnable
    affine transformation followed by tanh and a four-way softmax, and the
    resulting weights are row-wise L2 normalised (Eq. (32) of the paper). Every
    embedding is then re-weighted by its own node-wise weight and the four are
    summed to the aligned cross-view representation B_F (Eq. (33)).
    """

    def __init__(self, output_dim):
        super(MSC, self).__init__()
        self.mlp = Linear(output_dim * 4, 4)

    def forward(self, input_feature1, input_feature2, input_feature3, input_feature4, adj=None):
        features = [input_feature1, input_feature2, input_feature3, input_feature4]
        cat_features = torch.cat(features, 1)

        scores = torch.softmax(torch.tanh(self.mlp(cat_features)), dim=1)
        weights = F.normalize(scores, p=2, dim=1)

        fusion_features = 0
        for k, feature in enumerate(features):
            fusion_features = fusion_features + weights[:, k:k + 1] * feature
        return fusion_features
