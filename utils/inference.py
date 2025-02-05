import torch 
import os 
from matplotlib import pyplot as plt
import numpy as np 
import wfdb
from tqdm import tqdm
import json

from utils.text_to_emb import get_text_embedding
from utils.ecgplot import plot

def generation_from_net(diffused_model, net, batch_size, device, text_embed, condition, num_channels=4, dim=128):
    net.eval()
    xi = torch.randn(batch_size, num_channels, dim)
    xi = xi.to(device)
    timesteps = tqdm(diffused_model.timesteps)
    for _, i in enumerate(timesteps):
        t = i*torch.ones(batch_size, dtype=torch.long)
        with torch.no_grad():

            # change this line to fit your unet 
            if condition:
                noise_predict = net(xi, t, text_embed, condition)
            else:
                noise_predict = net(xi, t, text_embed)

            xi = diffused_model.step(model_output=noise_predict, 
                                     timestep=i, 
                                     sample=xi)['prev_sample']
    return xi 

def batch_generate_ECG(settings, 
                       unet, 
                       diffused_model, 
                       decoder, 
                       condition):

    save_path = settings['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_img = settings['save_img']
    if not save_img:
        print("Ignore image drawing and saving...")

    text = settings['text']
    gender = settings['gender']
    age = settings['age']
    hr = settings['hr']
    batch = settings['gen_batch']

    features_file_content = {}

    print('Diagnosis: The report of the ECG is that {' + text + '}.')
    text_embed = get_text_embedding(text)

    text_embed = np.array(text_embed)
    text_embed = np.repeat(text_embed[np.newaxis, :], 1, axis=0)
    text_embed = np.repeat(text_embed[np.newaxis, :, :], batch, axis=0)

    text_embed = torch.Tensor(text_embed)
    device = torch.device(settings['device'] if torch.cuda.is_available() else "cpu")
    text_embed = text_embed.to(device)
    
    verbose = settings['verbose']
    if verbose:
        print(text_embed.shape)

    features_file_content.update({"batch": batch}) 
    features_file_content.update({"Diagnosis": text}) 

    condition_dict = None
    if condition:
        condition_dict = {'gender': gender, 'age': age, 'heart rate': hr}
        for key in condition_dict:
            features_file_content.update({key: condition_dict[key]}) 
        condition_dict['gender'] = 1 if gender == 'M' else 0

        for key in condition_dict:
            condition_dict[key] = np.array([condition_dict[key]])
            condition_dict[key] = np.repeat(condition_dict[key][np.newaxis, :], 1, axis=0)
            condition_dict[key] = np.repeat(condition_dict[key][np.newaxis, :], batch, axis=0)
            if verbose:
                print(condition_dict[key].shape)
            condition_dict[key] = torch.Tensor(condition_dict[key])
            condition_dict[key] = condition_dict[key].to(device)
        if verbose:
            print(condition_dict)

    unet.to(device) 
    decoder.to(device)
    latent = generation_from_net(diffused_model, unet, batch_size=batch, device=device, text_embed=text_embed, condition=condition_dict)
    

    if save_img:
        gen_ecg = decoder(torch.Tensor(latent))
        for j in range(batch):
            output = gen_ecg[j]

            output_ = output.squeeze(0).detach().cpu().numpy()
            plot(output_.transpose(1, 0), sample_rate = 102.4, title = None, columns = 1, row_height = 3)
            plt.savefig(os.path.join(save_path , f'{j} Generated ECG.png'))
            plt.close()

    with open(os.path.join(save_path , 'features.json'), 'w') as json_file:
        json.dump(features_file_content, json_file, indent=4)
        print(f"Features has been successfully written to {save_path}/features.json")
        