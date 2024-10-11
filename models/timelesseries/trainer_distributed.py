import os 
from typing import Iterator, List, Optional 
from operator import itemgetter

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, Sampler, DistributedSampler


'''
world_size is the total number of processes (in general one per gpu)
rank is the number of the current node (goes from 0 to tot_num_of_nodes)
rank is only useful is we train over multiple nodes
'''

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def spawn_processes(train_fn, world_size):
    # spawn will automatically pass the index of the process as first arg to train_fn
    torch.multiprocessing.spawn(train_fn, args=(world_size,), nprocs=world_size, join=True)


class DistributedModel(DistributedDataParallel):
    
    def init_stats(self):
        self.module.init_stats()

    def reset_stats(self):
        self.module.reset_stats()

    def train_stats(self):
        return self.module.train_stats()

    def valid_stats(self):
        return self.module.valid_stats()

    def train_step(self, data):
        return self.module.train_step(data)

    def valid_step(self, data):
        return self.module.valid_step(data)

    def forward(self, x):
        return self.module.forward(x)

    def define_loss_func(self):
        return self.module.define_loss_func()

    def define_optimizer(self):
        return self.module.define_optimizer()

    def define_oscheduler(self):
        return self.module.define_scheduler()






