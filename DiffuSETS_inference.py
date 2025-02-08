import torch
import argparse 
import json
from diffusers import DDPMScheduler
from unet.unet_conditional import ECGconditional
from utils.inference import batch_generate_ECG 
from vae.vae_model import VAE_Decoder

def parse_arg():
    parser = argparse.ArgumentParser(description='DiffuSETS Inference') 
    parser.add_argument('--text', 
                        help="Clinical text report, multi-reports should be split by '|' ",
                        default="Sinus rhythm|Normal ECG.")
    parser.add_argument('--age', 
                        type=int, 
                        help="Age of patient",
                        default=50)
    parser.add_argument('--sex', 
                        type=str, 
                        help="Sex of patient",
                        default="M")
    parser.add_argument('--hr', 
                        type=int, 
                        help="Heart Rate of patient",
                        default=80)

    args = parser.parse_args()
    return args

def main(text: str="Sinus rhythm|Normal ECG.", age: int=50, hr: int=80, sex: str="M"):

    assert sex in ["F", "M"], "Sex should be either 'F' or 'M'!"
    assert age > 0 and age < 110
    assert hr > 0

    unet = ECGconditional(number_of_diffusions=1000, kernel_size=7, num_levels=7, n_channels=4, text_embed_dim=8192)

    unet_path = "./prerequisites/unet_model.pth"
    unet.load_state_dict(torch.load(unet_path, map_location='cpu'))

    diffused_model = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.0120)
    diffused_model.set_timesteps(1000)

    decoder = VAE_Decoder()
    vae_path = "./prerequisites/vae_model.pth" 
    checkpoint = torch.load(vae_path, map_location='cpu')
    decoder.load_state_dict(checkpoint['decoder'])

    condition = {"text": text,
                 "age": age,
                 "sex": sex,
                 "hr": hr}

    batch_generate_ECG(unet=unet, 
                       diffused_model=diffused_model, 
                       decoder=decoder, 
                       condition=condition)


if __name__ == "__main__":
    args = parse_arg()
    main(text=args.text, age=args.age, hr=args.hr, sex=args.sex) 
