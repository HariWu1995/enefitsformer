from copy import deepcopy
from typing import Dict, Iterable

import numpy as np
import torch
from torch import nn


class FeaturesNormalization(nn.Module):
    
    def __init__(self, features_dim: int, dropout_rate: float = 0.369, 
                         hidden_dim: int = None, instance_norm: bool = False):
        super().__init__()
        if not hidden_dim:
            hidden_dim = features_dim
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.regularize = nn.Dropout(p=dropout_rate)
        self.interconn = nn.Linear(in_features=self.features_dim, 
                                  out_features=self.hidden_dim, bias=True)
        self.activate_ = nn.Mish()
        self.normalize = nn.InstanceNorm1d(num_features=self.hidden_dim) if instance_norm \
                       else nn.BatchNorm1d(num_features=self.hidden_dim)

    def forward(self, features: torch.Tensor):
        features = torch.swapaxes(features, axis0=1, axis1=2)
        features = self.normalize(features)  # Batch / Instance-Norm1D applied on axis=1
        features = torch.swapaxes(features, axis0=1, axis1=2)
        features = self.interconn(features)  # shape: (B, L, Dh)
        features = self.activate_(features)
        features = self.regularize(features)
        return features


class FeaturesAggregation(nn.Module):
    
    def __init__(self, schema: Iterable[Dict], include_target: bool = False,
                                                is_timeseries: bool = True, **kwargs):
        super().__init__()
        self.Embedders = nn.ModuleDict()
        self.features_dim = 0
        self.features_order = list()
        self.features_schema = dict()
        self.is_targets = list()
        self.is_timeseries = is_timeseries

        for f_info in deepcopy(schema):
            f_is_target = f_info.get('is_target', False)
            if (not include_target) and f_is_target:
                continue

            f_name = f_info['name']
            if f_info['type'] == 'categorical':
                n_classes = f_info['n_classes'] + 1
                emb_dim = f_info.get('embedding_dim', 1+int(np.log(n_classes)))
                self.features_dim += emb_dim
                self.Embedders[f_name] = nn.Embedding(num_embeddings=n_classes, 
                                                      embedding_dim=emb_dim)
                self.is_targets.extend([f_is_target] * emb_dim)

            elif f_info['type'] == 'multilabel':
                n_classes = f_info['n_classes']
                self.features_dim += n_classes
                self.is_targets.extend([f_is_target] * emb_dim)

            elif f_info['type'] in ['numerical','boolean','ordinal']:
                self.features_dim += 1
                self.is_targets.append(f_is_target)

            self.features_order += [f_name]
            self.features_schema[f_name] = f_info
            self.features_schema[f_name].pop('name')

    def forward(self, tensors: Dict[str, torch.Tensor]):

        features = []
        for feat in self.features_order:
            if self.features_schema[feat]['type'] == 'categorical':
                feat_tensor = self.Embedders[feat](tensors[feat].int())  # (B, L) -> (B, L, D)
            elif self.features_schema[feat]['type'] == 'multilabel':
                feat_tensor = tensors[feat]      # require multi-hot tensor of shape (B, L, D)
            elif self.features_schema[feat]['type'] == 'numerical':
                feat_tensor = torch.unsqueeze(tensors[feat], dim=2)      # (B, L) -> (B, L, 1)
            features.append(feat_tensor)
        features = torch.cat(features, dim=2).to(dtype=torch.float16)
        if not self.is_timeseries:
            features = features.squeeze(dim=1)
        return features.to(dtype=torch.float)


