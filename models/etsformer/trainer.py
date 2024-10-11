"""
Reference:
    https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html
"""

import os 
import json
import copy
import random as rd
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import trainer_helper as helper
from . import trainer_distributed as distributed
from .callbacks import EarlyStopping


class TimeSeriesTrainer(Dataset):

    def __init__(self, model: nn.Module, train_dataset: Dataset, valid_dataset: Dataset, 
                    loss_fn: str='mse', optimizer: str='adam', lr_scheduler: str='reduce',
                    lr: float=0.1, num_epochs: int=1, num_steps_forecast: int=5, reproducibity_seed: int=0, 
                    device=None, distributed: bool=False, verbose: bool=True, patience: int=0,
                    results_path: str=None, checkpoint_path: str=None, summary_args: dict={}, 
                    tb_logs: bool=False, tb_checkpoint_rate: int=0, tb_embeddings_num: int=0,):
        """
        Parameters 
        __________
        model: instance of torch.nn.Module
        train_dataset (torch.utils.data.Dataset): dataset for training
        valid_dataset (torch.utils.data.Dataset): dataset for validation 
        num_epochs (int): number of epochs for training
        seed (int): random seed used for reproducibility
        device: device on which perform training, eg: cpu, cuda:0, cuda:1, ...
                If device is None and GPUs are detected on the machine, 
                    the one with the lower memory usage will be choosen. 
                default=None
        distributed (bool): allow distributed training on multiple gpus, 
                            using the distributed module of pytorch on a single node. 
                            default=False
        verbose (bool): print info on console at every step. If False, prints only at the end of each epoch. 
                        default=True
        results_path (str): absolute path to the folder in which tensorboard logs, model, checkpoint, ecc... will be stored. 
                            default=None
        tb_logs (bool): If true, tensorboard will be used to store statistics at each epoch, and the final model at the end of the training. 
                        The path for the tb logs will be something like this: 
                           - results_path/model_name/dd_mm_aa_hh:mm:ss/
                        At the end of the training, you will find inside all the info for tensorboard and the final model state dict (model_name.pt). 
                        If tb_checkpoint_rate is greater than 0, also a folder with checkpoints will be created.
                        default=False
        tb_checkpoint_rate (int): The number of epochs at which checkpoint are stored, 
                                  Eg: 2 -> every 2 epochs, 
                                      3 -> every 3 epochs, ... . 
                                  default=0
        checkpoint_path (str): absolute path to the checkpoint file that you want to load before training. 
                                This can be used to start the training from a previous computed checkpoint. 
                                We expect a dictionary with these keys:  
                                - 'model_state_dict': dict, required
                                - 'epoch': int, optional 
                                - 'optimizer_state_dict': dict, optional
                                - 'loss': train_loss, optional
                               default=None
        """
        # model info
        self.model      = model
        self.model_name = model.name
        self.num_steps_forecast = num_steps_forecast

        # data info
        self.train_dataset = train_dataset 
        self.valid_dataset = valid_dataset

        # train info
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self._multi_train = False
        self._conf_num = 0 

        # device info
        self.gpu_num = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device
        self.model._device = device

        # initial stats on the model
        self.model.init_stats()

        # initialize the early_stopping object
        if patience > 0:
            self.early_stopper = EarlyStopping(patience=patience, path=checkpoint_path, verbose=verbose)
        else:
            self.early_stopper = None

        # args 
        self.args = summary_args
        self.print_stats = verbose

        # tensorboard
        self.tb_logs = tb_logs
        self.save_path = results_path
        self.tb_embeddings_num = tb_embeddings_num
        self.tb_embeddings = self.tb_embeddings_num > 0
        self.load_path = None
        if tb_checkpoint_rate > 0:
            self.save_checkpoint = True
            self.checkpoint_rate = tb_checkpoint_rate
        else: 
            self.save_checkpoint = False

        if distributed and (self.gpu_num < 2):
            raise ValueError(f"cannot use distributed training, only {self.gpu_num} gpu are present")

        # set true only if distributed is enable and device is not set to any particular one
        # at this point we are also sure to have at least 2 gpus for training
        self._distributed = (distributed and (self.device is None))

        # tb checks
        if tb_logs and self.save_path is None: 
            raise ValueError("results_path should be defined if tb_logs is True")
        if self.tb_embeddings and not tb_logs: 
            raise ValueError("tb_logs should be True if tb_embeddings_num is greater than 0")

        # checkpoint checks
        if tb_checkpoint_rate > 0 and not tb_logs: 
            raise ValueError("tb_logs should be True if tb_checkpoint_rate is greater than 0")

        # set random seed for reproducibity
        rd.seed(reproducibity_seed)
        torch.manual_seed(reproducibity_seed)

        if self._distributed:
            helper.print_summary(self.model, None, self._distributed, self.args, self._multi_train, self._conf_num) 
        else:
            device = self._check_device(self.device)
            helper.print_summary(self.model, device, self._distributed, self.args, self._multi_train, self._conf_num) 

    def _check_device(self, device):
        if device is None:
            return torch.device('cuda:'+str(helper.get_free_gpu()) if torch.cuda.is_available() else 'cpu')
        return device

    def _send_to_device(self, data: dict, device):
        data.update({k: d.to(device) for k, d in data.items()})

    def _tb_setup_tensorboard(self):
        pass

    def _save_checkpoint(self, epoch, model, optimizer, scheduler, train_loss, last=False):
        name = self.model_name
        name += '.pt' if last else f'_ckpt_epoch_{epoch}.pt'
        path = os.path.join(self.tb_logdir, 'checkpoints', name)

        torch.save({
                           'epoch' :         epoch,
                            'loss' :      train_loss,
                'model_state_dict' :     model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
        }, path)

    def _save_epoch_stats(self, epoch, train_len, valid_len, train_stats, valid_stats, last_lr):
        for key, value in train_stats.items():
            self.tb_writer.add_scalar('Train/{}'.format(key), value / train_len, epoch)

        for key, value in valid_stats.items():
            self.tb_writer.add_scalar('Valid/{}'.format(key), value / valid_len, epoch)

        self.tb_writer.add_scalar('Learning rate', last_lr, epoch)

    def _print_epoch_stats(self, pbar, curr_lr, losses, stats, end: bool = False):

        out_dict = dict()
        out_dict.update({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()})
        out_dict.update({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in stats.items()})
        out_dict.update({'lr': curr_lr})

        helper.print_epoch_stats(pbar, end, **out_dict)

    def train(self):
        if self._distributed:
            distributed.spawn_processes(self._train_loop, self.gpu_num)
        else:
            device = self._check_device(self.device)
            self._train_loop(device, 0)

    def _train_loop(self, gpu, world_size):
        # gpu can either be a device or index of device in case of distributed training
        if torch.cuda.is_available():
            device = torch.device(gpu)
            torch.cuda.set_device(gpu)
        else:
            device = self.device

        # use a deep copy of the model on each sub-process in case of distributed training
        if self._distributed:
            model = copy.deepcopy(self.model)
        else:
            model = self.model

        # this is done in place
        model.to(device)
        model._device = device
        model.define_optimizer(self.optimizer, lr=self.lr)
        model.define_scheduler(self.lr_scheduler)

        # distributed=false -> master=True (only one device that is the master)
        # distributed=True  -> (multiple gpus, only cuda:0 is the master)
        master = ((not self._distributed) or device.index == 0)

        # num of batches
        train_len = len(self.train_dataset)
        valid_len = len(self.valid_dataset)
        optimizer = model.optimizer
        scheduler = model.scheduler

        if self.tb_logs and master:
            self._tb_setup_tensorboard()
            # TODO: get the input shape and save the model graph
            # input_sample = next(iter(train_loader))
            # self._tb_save_graph(model, input_sample.shape)

        # load checkpoint
        epoch_start = 0
        if self.load_path:
            ckpt_dict = torch.load(self.load_path, map_location=device)
            model.load_state_dict(ckpt_dict['model_state_dict'])
            if 'optimizer_state_dict' in ckpt_dict:
                model.optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt_dict:
                scheduler.load_state_dict(ckpt_dict['scheduler_state_dict'])
            if 'epoch' in ckpt_dict:
                epoch_start = ckpt_dict['epoch'] + 1

        # trackers of training / validation losses
        all_train_losses = []
        all_valid_losses = []

        # training loop
        for epoch in range(epoch_start, epoch_start+self.num_epochs):
            print(f"Epoch {epoch} / {self.num_epochs}")

            model.reset_stats()
            model.train()

            train_dataset, train_losses = self.train_dataset, {}
            valid_dataset, valid_losses = self.valid_dataset, {}

            # see https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
            # if self._distributed:
            #     train_sampler.set_epoch(epoch)
            #     valid_sampler.set_epoch(epoch)

            # if verbose use tqdm to print status bar
            loader = range(train_len)
            if self.print_stats and master:
                loader = tqdm(loader, bar_format='{n_fmt}/{total_fmt} |{bar:4}|{elapsed}{postfix}',
                                      unit_scale=True, leave=True, dynamic_ncols=True)
            for i in loader:
                data = train_dataset[i]

                # send to device
                for d in data:
                    self._send_to_device(d, device)

                # reset the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # training step
                losses = model.train_step(*data, self.num_steps_forecast)
                loss = losses.pop('total')

                # backward + optimize
                loss.backward()
                optimizer.step()

                # record training loss
                for k, v in losses.items():
                    if i == 0:
                        train_losses[k] = [v.item()]
                    else:
                        train_losses[k] += [v.item()]

                # (print_stats && (!distributed or device == gpu:0))
                if self.print_stats and master:
                    self._print_epoch_stats(loader, model.get_latest_lr(), losses, model.train_stats())

            if self.print_stats and master:
                losses = {k: np.mean(v) for k, v in train_losses.items()}
                self._print_epoch_stats(loader, model.get_latest_lr(), losses, model.train_stats(), end=True)

            # if verbose use tqdm to print status bar
            loader = range(valid_len)
            if self.print_stats and master:
                loader = tqdm(loader, bar_format='{n_fmt}/{total_fmt} |{bar:4}|{elapsed}{postfix}',
                                      unit_scale=True, leave=True, dynamic_ncols=True)
            model.eval()
            with torch.no_grad():
                for i in loader:
                    data = valid_dataset[i]

                    # send to device
                    for d in data:
                        self._send_to_device(d, device)

                    # valid step
                    val_losses = model.valid_step(*data, self.num_steps_forecast)
                    for k, v in val_losses.items():
                        if i == 0:
                            valid_losses[k] = [v.item()]
                        else:
                            valid_losses[k] += [v.item()]

                    if self.print_stats and master:
                        _ = val_losses.pop('total')
                        self._print_epoch_stats(loader, 0., val_losses, model.valid_stats())

                if self.print_stats and master:
                    val_losses = {k: np.mean(v) for k, v in valid_losses.items() if k != 'total'}
                    self._print_epoch_stats(loader, 0., val_losses, model.valid_stats(), end=True)

            # lr decay step
            val_loss = np.mean(valid_losses['total'])
            if scheduler is not None:
                try:
                    scheduler.step()
                except TypeError:
                    scheduler.step(val_loss)    # for Reduce LR on Pleateau

            # run callbacks
            if self.early_stopper:
                self.early_stopper(val_loss, model)
                if self.early_stopper.early_stop:
                    print("Early stopping")
                    break

            if master and self.tb_logs:
                self._save_epoch_stats(epoch, train_len = train_len, 
                                              valid_len = valid_len,
                                            train_stats = model.train_stats(), 
                                            valid_stats = model.valid_stats(), 
                                                last_lr = model.get_latest_lr())

            # save checkpoint
            if master and self.save_checkpoint and (epoch > 0) and not (epoch % self.checkpoint_rate):
                self._save_checkpoint(epoch, model, optimizer, scheduler, loss)

        # train end
        helper.print_end_train()

        # tensorboard for saving results and embeddings
        if master and self.tb_logs:
            self._tb_save_results(model=model, train_len=train_len, valid_len=valid_len,
                                  epoch=epoch, optimizer=optimizer, scheduler=scheduler, train_loss=loss)

        if master and self.tb_embeddings:
            self._tb_save_embeddings(model, device, valid_loader)

    def multi_train(self, train_config_path):
        with open(train_config_path) as f:
            train_configs = json.load(f)
        # train_configs = pd.read_csv(train_config_path, sep=' ').to_dict(orient='records')

        # save initial model parameters 
        self._initial_model_param = copy.deepcopy(self.model.state_dict())
        self._multi_train = True

        helper.print_overall_summary(self.model, train_configs) 
        for i, configs in enumerate(train_configs):
            # set model attributes for the current config
            self.args.update(configs)
            for k, d in configs.items(): 
                setattr(self.model, k, d)

            # train on the current config
            self._conf_num = i
            self.train()

            # reset the parameters of the model
            self.model.reset_stats()
            self.model.load_state_dict(self._initial_model_param)
            torch.cuda.empty_cache()



