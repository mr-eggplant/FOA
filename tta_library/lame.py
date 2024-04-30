import torch
import torch.jit
import logging
from typing import List, Dict

import time
import torch.nn.functional as F
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class AffinityMatrix:

    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, sigma: float, **kwargs):
        self.sigma = sigma
        self.k = kwargs['knn']

    def __call__(self, X):

        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        # mask = torch.eye(X.size(0)).to(X.device)
        # rbf = rbf * (1 - mask)
        return rbf


class linear_affinity(AffinityMatrix):

    def __call__(self, X: torch.Tensor):
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())

class LAME(nn.Module):
    """
    Our proposed method based on Laplacian Regularization.
    """  
    def __init__(self, model:VisionTransformer, knn=5, sigma=1.0, affinity='kNN', force_symmetry=True):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        self.model = model

        self.knn = knn
        self.sigma = sigma
        self.affinity = eval(f'{affinity}_affinity')(sigma=self.sigma, knn=self.knn)
        self.force_symmetry = force_symmetry

    def forward(self, x):
        with torch.no_grad():
            feats = self.model.forward_features(x)
            out = self.model.forward_head(feats)
            if self.imagenet_mask is not None:
                outputs = outputs[:, self.imagenet_mask]
            probas = out.softmax(dim=1)
            # --- Get unary and terms and kernel ---
            unary = - torch.log(probas + 1e-10)  # [N, K]
            
            feats = feats[:,0]   # [N, d]
            feats = F.normalize(feats, p=2, dim=-1)  # [N, d]
            kernel = self.affinity(feats)  # [N, N]
            if self.force_symmetry:
                kernel = 1/2 * (kernel + kernel.t())

            # --- Perform optim ---
            Y = laplacian_optimization(unary, kernel)
        return Y
    
    def reset(self):
        pass

def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):

    E_list = []
    oldE = float('inf')
    Y = (-unary).softmax(-1)  # [N, K]
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        exponent = -unary + pairwise
        Y = exponent.softmax(-1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)

        if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
            # print(f'Converged in {i} iterations')
            break
        else:
            oldE = E

    return Y


def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E