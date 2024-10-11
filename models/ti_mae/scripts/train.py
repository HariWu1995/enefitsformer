import os
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

import hydra
import numpy as np
import pandas as pd

from ti_mae.src.nn.model_lit import LitAutoEncoder
from src.data.dataset import HourParquetDataset

from torch.utils.data import DataLoader
from torchinfo import summary
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


@hydra.main(version_base=None, config_path="/workspace/ti-mae/configs", config_name="base.yaml")
def main(cfg):
    clip_values = pd.read_parquet(cfg.data.clip_path)
    autoencoder = LitAutoEncoder(**cfg.model,weights=np.array([1.,1.,1.]))
    autoencoder._set_hparams(cfg)
    summary(autoencoder.model,(((1,cfg.model.in_chans,cfg.model.seq_len))))

    data_path = cfg.data.path
    paths = []
    for p,_,fs in os.walk(data_path):
        for f in fs:
            if f.endswith('parquet') and ('clip' not in f):
                paths.append(os.path.join(p,f))
    paths = sorted(paths,key= lambda x : x.split('/')[-1])

    eval_paths = paths[-cfg.data.eval_dataset.size:]
    paths      = paths[:-cfg.data.eval_dataset.size]
    paths      = paths[-cfg.data.train_dataset.size:]
    
    print(f'{str(datetime.now())} : Creating train dataset.')
    train_dataset = HourParquetDataset(paths, **cfg.data, clip_values=clip_values)
    print(f'{str(datetime.now())} : Train dataset size : {len(paths)} hours , {len(train_dataset)} samples.')

    print(f'{str(datetime.now())} : Creating eval dataset.')
    eval_dataset = HourParquetDataset(eval_paths,
                                      stats=[train_dataset.stats[-1]], 
                                      clip_values=clip_values,
                                      mode='eval',
                                      **cfg.data)
    print(f'{str(datetime.now())} : Eval dataset size : {len(eval_paths)} hours , {len(eval_dataset)} samples.')

    autoencoder = LitAutoEncoder(**cfg.model,weights=train_dataset.weights)
    autoencoder._set_hparams(cfg)
    
    # weights = load('/workspace/ti-mae/nan_weight{batch_idx}.pt')
    # autoencoder.load_state_dict({'model.' + k : v.cpu() for k,v in weights.items()})

    train_loader = DataLoader(train_dataset, cfg.data.train_batch_size,
                                 num_workers=cfg.data.loader_workers,
                                     shuffle=cfg.data.train_dataset.shuffle)
    eval_loader = DataLoader(eval_dataset,  cfg.data.eval_batch_size,
                                num_workers=cfg.data.loader_workers,)

    callbacks = [ModelCheckpoint(**cfg.callbacks.checkpointing),
                   EarlyStopping(**cfg.callbacks.early_stopping)]

    trainer = pl.Trainer(callbacks=callbacks, **cfg.trainer)
    trainer.fit(autoencoder,train_loader,eval_loader,ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()