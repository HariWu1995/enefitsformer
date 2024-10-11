from typing import Callable, Any
from functools import partial
from collections import defaultdict

import torch


def _get_zero_tensor(device):
    return torch.tensor(0., device=device, requires_grad=False)


def _default_stats(device):
    default_callable = partial(_get_zero_tensor, device=device) 
    return defaultdict(default_callable)


class ModelTemplate(torch.nn.Module):
    ''' 
    Model class that should be inhereted for using the Trainer.
    Children of the torch.nn.Module, so the user should also define the forward pass.
    Train step and Valid step are used to define the training and validation logic.
    In the training and valid step, we can store values like loss and evaluation metrics by using the save_epoch_stats
    '''
    def __init__(self, name):
        super().__init__()
        self._device = None
        self.name = name

    def _train_step_unimplemented(self, *input: Any):
        raise NotImplementedError

    def _valid_step_unimplemented(self, *input: Any):
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any):
        raise NotImplementedError

    def _define_loss_func_unimplemented(self, *input: Any):
        raise NotImplementedError

    def _define_optimizer_unimplemented(self, *input: Any):
        raise NotImplementedError

    def _define_scheduler_unimplemented(self, *input: Any):
        raise NotImplementedError

    def save_train_stats(self, **kwargs):
        self._train_stats.update({k: self._train_stats[k] + d for k, d in kwargs.items()})

    def save_valid_stats(self, **kwargs):
        self._valid_stats.update({k: self._valid_stats[k] + d for k, d in kwargs.items()})

    def init_stats(self):
        self._train_stats = _default_stats(self._device)
        self._valid_stats = _default_stats(self._device) 

    def reset_stats(self):
        self._train_stats.clear()
        self._valid_stats.clear()

    def train_stats(self):
        return self._train_stats

    def valid_stats(self):
        return self._valid_stats

    '''
    These methods should be overrided by the user model class, otherwise throw a NotImplemented exception.
    '''
    train_step: Callable[..., Any] = _train_step_unimplemented  # should return the value of the loss function
    valid_step: Callable[..., Any] = _valid_step_unimplemented  # might return the value of the loss, not important

    forward: Callable[..., Any] = _forward_unimplemented  # defines the forward pass for getting in output embeddings
    define_optimizer: Callable[..., Any] = _define_optimizer_unimplemented
    define_scheduler: Callable[..., Any] = _define_scheduler_unimplemented
    define_loss_func: Callable[..., Any] = _define_loss_func_unimplemented


