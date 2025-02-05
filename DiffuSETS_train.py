import argparse 
import json
import logging
import os
import torch 
import pandas as pd 
from torch.utils.data import DataLoader 
from diffusers import DDPMScheduler
from dataset.mimic_iv_ecg_dataset import DictDataset 

from unet.unet_conditional import ECGconditional 
from utils.train import train_model 

def parse_arg():
    parser = argparse.ArgumentParser(description='DiffuSETS Training') 
    parser.add_argument('config', help='Root of training configuration')

    args = parser.parse_args()
    return args

def main():
    args = parse_arg() 

    with open(args.config, 'r') as f:
        config = json.load(f) 

    meta = config['meta']
    roots = config['dependencies']
    h_ = config['hyper_para']

    k_max = 0
    if not os.path.exists(roots['checkpoints_dir']):
        os.makedirs(roots['checkpoints_dir'])
    for item in os.listdir(roots['checkpoints_dir']):
        if meta['exp_type'] + "_" in item:
            k = int(item.split('_')[-1]) 
            k_max = k if k > k_max else k_max
    save_weights_path = os.path.join(roots['checkpoints_dir'], f"{meta['exp_type']}_{k_max + 1}")

    try:
        os.makedirs(save_weights_path)
    except:
        pass

    logger = logging.getLogger(f"{meta['exp_type']}_{k_max + 1}")
    logger.setLevel('INFO')
    fh = logging.FileHandler(os.path.join(save_weights_path, 'train.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(meta)
    logger.info(h_)

    train_dataset = DictDataset(roots['dataset_path']) 
    train_dataloader = DataLoader(train_dataset, batch_size=h_['batch_size'])

    n_channels = 4 
    unet = ECGconditional(h_['num_train_steps'], kernel_size=h_['unet_kernel_size'], num_levels=h_['unet_num_level'], n_channels=n_channels, text_embed_dim=8192)

    diffused_model = DDPMScheduler(num_train_timesteps=h_['num_train_steps'], beta_start=h_['beta_start'], beta_end=h_['beta_end'])


    train_model(meta=meta, 
                save_weights_path=save_weights_path, 
                dataloader=train_dataloader, 
                diffused_model=diffused_model, 
                unet=unet, 
                h_=h_, 
                logger=logger)

if __name__ == '__main__': 
    main()
