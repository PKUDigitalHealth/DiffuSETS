import torch
import argparse 
import json
from diffusers import DDPMScheduler
from unet.unet_conditional import ECGconditional
from utils.inference import batch_generate_ECG 
from vae.vae_model import VAE_Decoder

def parse_arg():
    parser = argparse.ArgumentParser(description='DiffuSETS Inference') 
    parser.add_argument('config', help='Root of training configuration')

    args = parser.parse_args()
    return args

def main():

    args = parse_arg() 

    with open(args.config, 'r') as f:
        config = json.load(f) 

    settings = config['inference_setting']
    h_ = config['hyper_para']

    settings['device'] = config['meta']['device']

    n_channels = 4
    unet = ECGconditional(h_['num_train_steps'], kernel_size=h_['unet_kernel_size'], num_levels=h_['unet_num_level'], n_channels=n_channels, text_embed_dim=8192)

    unet_path = settings['unet_path']
    unet.load_state_dict(torch.load(unet_path, map_location='cpu'))

    diffused_model = DDPMScheduler(num_train_timesteps=h_['num_train_steps'], beta_start=h_['beta_start'], beta_end=h_['beta_end'])
    diffused_model.set_timesteps(settings['inference_timestep'])


    decoder = VAE_Decoder()
    vae_path = config['dependencies']['vae_path']
    checkpoint = torch.load(vae_path, map_location='cpu')
    decoder.load_state_dict(checkpoint['decoder'])

    batch_generate_ECG(settings=settings, 
                    unet=unet, 
                    diffused_model=diffused_model, 
                    decoder=decoder, 
                    condition=True)


if __name__ == "__main__":
    main() 
