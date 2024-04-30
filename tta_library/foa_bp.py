"""
Copyright to FOA Authors ICML 2024
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from models.vpt import PromptViT

class FOA_BP(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self,
                model:PromptViT,
                optimizer,
                fitness_lambda = 30):
        super().__init__()
        self.optimizer = optimizer
        self.fitness_lambda = fitness_lambda

        self.model = model
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.hist_stat = None

    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean
            
    def _get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-768:] - self.hist_stat

    def forward(self, x):
        shift_vector = self._get_shift_vector()
        outputs, batch_mean = forward_and_get_loss(x, self.model, self.optimizer, self.fitness_lambda, self.train_info, shift_vector, self.imagenet_mask)
        self._update_hist(batch_mean[-768:])
        return outputs
    
    def obtain_origin_stat(self, train_loader):
        print('===> begin calculating mean and variance')
        self.model.eval()
        features = []
        with torch.no_grad():
            for _, dl in enumerate(train_loader):
                images = dl[0].cuda()
                feature = self.model.layers_cls_features(images)
                features.append(feature)
            features = torch.cat(features, dim=0)
            self.train_info = torch.std_mean(features, dim=0) # occupy 0.2MB 
        print('===> calculating mean and variance end')

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.hist_stat = None

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def copy_model_only(model):
    source_model = deepcopy(model)
    for param in source_model.parameters():
        param.detach_()
    return source_model

criterion_mse = nn.MSELoss(reduction='mean').cuda()

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_get_loss(images, model:PromptViT, optimizer, fitness_lambda, train_info, shift_vector, imagenet_mask):
    features = model.layers_cls_features_with_prompts(images)
    
    batch_std, batch_mean = torch.std_mean(features, dim=0)
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])
    # NOTE: $lambda$ should be 20 for ImageNet-R!!
    discrepancy_loss = fitness_lambda * (std_mse + mean_mse)

    cls_features = features[:, -768:]
    del features

    output = model.vit.head(cls_features)
    if imagenet_mask:
        output = output[:, imagenet_mask]
    entropy_loss = softmax_entropy(output).mean()
    loss = discrepancy_loss + entropy_loss

    with torch.no_grad():
        if shift_vector is not None:
            output = model.vit.head(cls_features + 1. * shift_vector)
            if imagenet_mask:
                output = output[:, imagenet_mask]
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return output, batch_mean

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names