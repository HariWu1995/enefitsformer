from typing import Dict, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.aggregation import FeaturesAggregation as FeatAggregator, \
                                FeaturesNormalization as FeatNormalizer
from .model_helper import ModelTemplate
from .model import ETSFormer as TsRegressor, ClassificationWrapper as TsClassifier


class MultiVariateForecast(ModelTemplate):

    def __init__(self, features_schema: Iterable[Dict],
                          model_schema: Dict[str, Dict],
                                  name: str = 'Multi-Variate Forecasting with Exponential-Smoothing Time-Series Transformer'):
        super().__init__(name)

        normalizer_schema = model_schema.pop("normalization")
        transformer_schema = model_schema.pop("transformation")
        predictions_schema = model_schema.pop("predictions")

        self.aggregate = FeatAggregator(schema=features_schema, include_target=True)
        self.normalize = FeatNormalizer(features_dim=self.aggregate.features_dim,
                                         hidden_dim=self.aggregate.features_dim, **normalizer_schema)
        self.transform = TsRegressor( time_features=self.aggregate.features_dim,
                                    feature_weights=self.aggregate.is_targets, **transformer_schema)

        # Define tasks and losses
        self.tasks = nn.ModuleDict()
        self.losses = dict()
        for pred_schema in predictions_schema:
            task_n = pred_schema.pop('name')
            target = pred_schema.pop('target')
            weight = pred_schema.pop('weight')

            self.losses[task_n] = {
                    'target' : target,
                    'weight' : weight,
                'train_only' : pred_schema.get('train_only', False),
                      'task' : pred_schema['task'],
                      'loss' : self.define_loss_func(**pred_schema)
            }

            if pred_schema['task'] == 'regression':
                self.tasks[task_n] = nn.Linear(in_features=self.aggregate.features_dim,
                                               out_features=1)
            else:
                assert isinstance(target, (list, tuple)), \
                                 "target must be either list or tuple"
                classes_weight = pred_schema.get('classes_weight', None)
                if classes_weight:
                    self.losses[task_n].update({'classes_weight': torch.tensor(classes_weight)})
                self.tasks[task_n] = nn.Sequential(
                                     nn.Linear(in_features=self.aggregate.features_dim,
                                               out_features=len(target)),
                                     nn.Sigmoid() if pred_schema.get('multilabel', False)
                                else nn.Softmax())

    def forward(self, x: Dict[str, torch.Tensor], forecast_horizon: int = 0):
        x = self.aggregate(x).to(dtype=torch.float)
        x = self.normalize(x)
        x = self.transform(x, forecast_horizon=forecast_horizon)

        y = dict()
        for task, predict in self.tasks.items():
            y[task] = predict(x)
        return y

    def train_step(self, x, y, forecast_horizon: int):
        return self.loss_step(x, y, forecast_horizon)

    def valid_step(self, x, y, forecast_horizon: int):
        return self.loss_step(x, y, forecast_horizon, training=False)

    def loss_step(self, x, y, forecast_horizon: int, training: bool = True):
        y_hat = self.forward(x, forecast_horizon=forecast_horizon)
        losses = dict(total=0.)
        for task, task_def in self.losses.items():
            if (task_def['train_only']) and (not training):
                continue

            task_yhat = y_hat[task]

            if task_def['task'] == 'classification':
                task_y = [torch.tensor(y[v]).unsqueeze(dim=2) for v in task_def['target']]
                task_y = torch.cat(task_y, dim=2).to(task_yhat.device)
                task_w = task_def.get('classes_weight', None)
                if task_w is not None:
                    task_w = task_w.to(task_yhat.device)
                task_loss = task_def['loss'](task_yhat, task_y, reduction='none', weight=task_w)

            elif task_def['task'] == 'regression':
                task_y = torch.tensor(y[task_def['target']])
                task_y = task_y.unsqueeze(dim=2).to(task_yhat.device)
                task_loss = task_def['loss'](task_yhat, task_y, reduction='none')

            task_loss = task_loss.mean()
            losses[task] = task_loss
            losses['total'] += task_loss * task_def.get('weight', 1.)
        return losses

    def loss_step_(self, x, y, forecast_horizon: int):
        y_hat = self.forward(x, forecast_horizon=forecast_horizon)
        y = self.aggregate(y)
        if isinstance(self.output_weights, torch.Tensor):
            loss = self.loss_fn(y_hat, y, reduction='none').to(self.output_weights.device)
            loss = torch.mul(loss.mean(dim=(0, 1)), self.output_weights)
            loss = loss.sum() / self.output_weights.sum()
        else:
            loss = self.loss_fn(y_hat, y, reduction='mean')
        return loss

    def define_loss_func(self, task: str, loss: str, **kwargs):
        loss = loss.lower()
        task = task.lower()

        if task == 'regression':
            if any(fn in loss for fn in ['msa', 'l1']):
                if 'smooth' in loss:
                    return F.smooth_l1_loss
                else:
                    return F.l1_loss
            elif 'huber' in loss:
                return F.huber_loss
            else:
                return F.mse_loss

        elif task == 'classification':
            if kwargs.get('multilabel', False):
                return F.binary_cross_entropy
            else:
                return F.cross_entropy

        else:
            raise ValueError(f"{task} is not supported!")

    def define_optimizer(self, optimizer: str = 'sgd', lr: float = 0.1, **kwargs):
        optimizer = optimizer.lower()
        if optimizer == 'adadelta':
            from torch.optim import Adadelta as Optimizer
        elif optimizer == 'adagrad':
            from torch.optim import Adagrad as Optimizer
        elif optimizer == 'adam':
            from torch.optim import Adam as Optimizer
        elif optimizer == 'adamw':
            from torch.optim import AdamW as Optimizer
        elif optimizer == 'nadam':
            from torch.optim import NAdam as Optimizer
        elif optimizer == 'radam':
            from torch.optim import RAdam as Optimizer
        elif optimizer == 'rmsprop':
            from torch.optim import RMSprop as Optimizer
        else:
            from torch.optim import SGD as Optimizer
        self.optimizer = Optimizer(self.parameters(), lr=lr, **kwargs)

    def define_scheduler(self, scheduler: str = 'reduce', **kwargs):
        if scheduler is None:
            return None
        scheduler = scheduler.lower()
        if scheduler.startswith('const'):
            from torch.optim.lr_scheduler import ConstantLR as LrScheduler
        elif scheduler.startswith('cosine'):
            if 'warm_restart' in scheduler:
                from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as LrScheduler
                if 'T_0' not in kwargs.items():
                    kwargs['T_0'] = 3
                if 'T_mult ' not in kwargs.items():
                    kwargs['T_mult '] = 3
            else:
                from torch.optim.lr_scheduler import CosineAnnealingLR as LrScheduler
                if 'T_max' not in kwargs.items():
                    kwargs['T_max'] = 7
            if 'eta_min' not in kwargs.items():
                kwargs['eta_min'] = 1e-7
        elif scheduler == 'multiplicative':
            from torch.optim.lr_scheduler import MultiplicativeLR as LrScheduler
            if 'lr_lambda' not in kwargs.items():
                kwargs['lr_lambda'] = lambda epoch: 0.95
        elif scheduler == 'multistep':
            from torch.optim.lr_scheduler import MultiStepLR as LrScheduler
            if 'milestones' not in kwargs.items():
                kwargs['milestones'] = [10, 25, 50, 100]
        elif scheduler == 'step':
            from torch.optim.lr_scheduler import StepLR as LrScheduler
            if 'step_size' not in kwargs.items():
                kwargs['step_size'] = 10
        elif scheduler == 'linear':
            from torch.optim.lr_scheduler import LinearLR as LrScheduler
        elif scheduler == 'exponential':
            from torch.optim.lr_scheduler import ExponentialLR as LrScheduler
            if 'gamma' not in kwargs.items():
                kwargs['gamma'] = 0.369
        # elif scheduler == 'chained':
        #     from torch.optim.lr_scheduler import ChainedScheduler as LrScheduler
        # elif scheduler == 'sequential':
        #     from torch.optim.lr_scheduler import SequentialLR as LrScheduler
        # elif scheduler == 'single_cycle':
        #     from torch.optim.lr_scheduler import OneCycleLR as LrScheduler
        elif scheduler == 'cyclic':
            from torch.optim.lr_scheduler import CyclicLR as LrScheduler
            if 'base_lr' not in kwargs.items():
                kwargs['base_lr'] = 0.01
            if 'max_lr' not in kwargs.items():
                kwargs['max_lr'] = 10 * kwargs['base_lr']
        elif scheduler == 'reduce':
            from torch.optim.lr_scheduler import ReduceLROnPlateau as LrScheduler
            if 'patience' not in kwargs.items():
                kwargs['patience'] = 3
            if 'factor' not in kwargs.items():
                kwargs['factor'] = 0.369
        else:
            return None

        if 'verbose' not in kwargs.items():
            kwargs['verbose'] = True
        self.scheduler = LrScheduler(optimizer=self.optimizer, **kwargs)

    def get_latest_lr(self):
        try:
            lr = self.scheduler.get_last_lr()[0]
        except AttributeError:
            lr = self.scheduler.optimizer.param_groups[0]['lr']
        return lr


