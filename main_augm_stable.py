#!/usr/bin/venv2thon

"""
Created on Thu Dec 19 14:50:43 2019
parametrized version that is working for server training

use of it: we have four modes:
train, test, infer and restore



@author: julkam
"""
import numpy as np
import utils 
import monai
import matplotlib
import yaml
import sys
from pathlib import Path
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import set_up
import set_inf
from box import Box


parser = argparse.ArgumentParser(description='Trainer.')
parser.add_argument('-c', '--config', dest='config', default='configuration_mri.yaml', help='configuration file')
parser.add_argument('-test', '--test', dest='test', default=False, action='store_true', help='test mode')
parser.add_argument('-train', '--train', dest='train', default=False, action='store_true', help='train mode')
parser.add_argument('-i', '--inference', dest='inference', default=False, action='store_true', help='inference mode')
parser.add_argument('-r', '--restore', dest='restore', default=False, action='store_true', help='restore and continue')
parser.add_argument('-w', '--weights', dest='weights', help='Checkpoint file')
parser.add_argument('-O', '--output-dir', dest='outdir', help='Output dir. Default: model/inference', default=None)
parser.add_argument('-in_dir', '--in_dir', dest='indir', help='Input dir for inference ', default=None)
parser.add_argument('volumes', metavar='volumes', type=str, nargs='*')

args = parser.parse_args()

#same as gui.py
params = yaml.safe_load(Path(args.config).read_text())

# report and save package versions
print('Python modules:')
params['packages'] = {}
for name,ver in utils.version((np, torch, pl, monai, sys, matplotlib)).items():
    print(f'{name}   : {ver}')
    params['packages'][name] = ver

if args.train or args.restore:
    # set up loggers
    logger = pl.loggers.TensorBoardLogger(**params['log'])
    configdir = Path(logger.log_dir)
    configdir.mkdir(parents=True, exist_ok=True)
    (configdir / 'checkpoints').mkdir()
    with open(configdir / 'configuration.yaml', 'w') as f:
        yaml.safe_dump(params, f)
    params['log']['configdir'] = configdir

# initialise the LightningModule
params = Box(params)

n_cl = params.data.num_class
print("-------------------------------------------checking batch---------------------", params['net'])
# net = unet_3D()
net = set_up.Framework(utils.class_loader(params['net'])[0], params, n_cl)
# default used by the Trainer
#early_stop = EarlyStopping('val_epoch_Loss', patience=10)
#checkpoint_callback = ModelCheckpoint(**params['checkpoint'], dirpath=str(configdir / 'checkpoints'),
#                                       filename='{epoch}-{val_epoch_Loss:.3f}')


if args.restore:
    print('checking the inference, and where the checkpoints were taken from', args.weights)
    x = torch.load(args.weights)
    weights=x['state_dict'] #loading weights from checkpoints
    checkpoint_callback = ModelCheckpoint(**params['checkpoint'], dirpath=str(configdir / 'checkpoints'),
                                          filename='{epoch}-{val_loss:.3f}')
    params['log']['configdir'] = Path(args.weights).parent.parent
    configdir = params['log']['configdir'] #to use it for restoring
    net.load_state_dict(weights)
    early_stop = EarlyStopping('val_epoch_Loss', patience=20)
    checkpoint_callback = ModelCheckpoint(**params['checkpoint'], dirpath=str(configdir / 'checkpoints'), filename='{epoch}-{val_epoch_Loss:.3f}')
    trainer = pl.Trainer(**params['lightning'], logger=logger, callbacks=[checkpoint_callback]) #early_stop

elif args.inference or args.test:
    print('checking the inference, and where the checkpoints were taken from', args.weights)
    x = torch.load(args.weights)
    weights=x['state_dict'] #loading weights from checkpoints
    params['log']['configdir'] = Path(args.weights).parent.parent
    configdir = params['log']['configdir'] #to use it for restoring
    net.load_state_dict(weights)
    trainer = pl.Trainer(**params['lightning'])
elif args.train:
    #early_stop = EarlyStopping('val_epoch_Loss', patience=20)
    #early_stop = EarlyStopping(**params['earlyStopping'])
    # set up loggers and checkpoints
    tb_logger = pl.loggers.TensorBoardLogger(save_dir= Path(configdir))
    #lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(**params['checkpoint'], dirpath=str(configdir / 'checkpoints'), filename='{epoch}-{val_loss:.3f}')
    net_callbacks = [checkpoint_callback]
    # trainer = pl.Trainer(
    #     gpus=[0],
    #     max_epochs=200,
    #     num_sanity_val_steps=1,
    #     logger=tb_logger,
    #     callbacks=net_callbacks,
    #     log_every_n_steps= 1
    # )

    trainer = pl.Trainer(**params['lightning'], logger=tb_logger, callbacks=net_callbacks ) #early_stop
else:
    print('choose the mode for the network')
    raise NotImplementedError

if args.train or args.restore:
    trainer.fit(net)
    print('clean the memory')
    torch.cuda.empty_cache()
    print('memory is clean')
elif args.test:
    net.eval()
    trainer.test(net)
elif args.inference:
    net.eval()
    if args.indir:
        print('trying to make an infarance', args.indir)
        set_inf.inference_save (net, str(args.indir), args.outdir, None, params)
    else:
        print('trying to make an infarance', args.volumes[1])
        set_inf.inference_save (net, None, args.outdir, str(args.volumes[1]), params)
else:
    print('you did not select the mode')
    raise NotImplementedError
