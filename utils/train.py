import torch
import torch.nn.functional as F
import time
import os
import numpy as np

def train_epoch_channels(dataloader, 
                         unet, 
                         diffused_model, 
                         condition, 
                         optimizer, 
                         scheduler,
                         device, 
                         number_of_repetition=1):
    loss_list = []
    unet.train()
    for _ in range(number_of_repetition):
        for data, label in dataloader:

            # (length_embedding, B) 
            text_embed = label['text_embed']
            text_embed = np.array(text_embed)
            text_embed = text_embed.transpose(1, 0) 
            text_embed = np.repeat(text_embed[:, np.newaxis, :], 1, axis=1)
            text_embed = torch.Tensor(text_embed)
            text_embed = text_embed.to(device)

            latent = data.to(device)

            # t = torch.randperm(diffused_model.config.num_train_timesteps-2)[:latent.shape[0]] + 1 
            # compatible with larger batch size
            t = torch.randint(1, diffused_model.config.num_train_timesteps - 1, (latent.shape[0],))

            noise = torch.randn(latent.shape, device=latent.device)
            xt = diffused_model.add_noise(latent, noise, t)

            xt = xt.to(device)
            t = t.to(device)
            noise = noise.to(device)

            if condition:
                gender = []
                age = label['age']
                hr = label['hr']
                condition_dict = {}
            
                for ch in label['gender']:
                    if ch == 'M':
                        gender.append(1)
                    else:
                        gender.append(0)
                gender = np.array(gender)
                gender = np.repeat(gender[:, np.newaxis], 1, axis=1)
                gender = np.repeat(gender[:, :, np.newaxis], 1, axis=2)
                gender = torch.Tensor(gender)
                gender = gender.to(device)
                condition_dict.update({'gender': gender})

                age = np.array(age)
                age = np.repeat(age[:, np.newaxis], 1, axis=1)
                age = np.repeat(age[:, :, np.newaxis], 1, axis=2)
                age = torch.Tensor(age)
                age = age.to(device)
                condition_dict.update({'age': age})

                hr = np.array(hr)
                hr = np.repeat(hr[:, np.newaxis], 1, axis=1)
                hr = np.repeat(hr[:, :, np.newaxis], 1, axis=2)
                hr = torch.Tensor(hr)
                hr = hr.to(device)
                condition_dict.update({'heart rate': hr})

                for key in condition_dict:
                    condition_dict[key] = condition_dict[key].to(device)

                noise_estim = unet(xt, t, text_embed, condition_dict)
            else: 
                noise_estim = unet(xt, t, text_embed)

            # Batchwise MSE loss 
            loss = F.mse_loss(noise_estim, noise, reduction='sum').div(noise.size(0))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return sum(loss_list) / len(loss_list)


def train_model(meta, 
                save_weights_path, 
                dataloader,  
                diffused_model, 
                unet, 
                h_, 
                logger):
 
    device = torch.device(meta['device'] if torch.cuda.is_available() else "cpu")
    unet = unet.to(device)
    optimizer = torch.optim.AdamW(params=unet.parameters(), lr=h_['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=h_['epochs']*len(dataloader), eta_min=0.1*h_['lr'])

    min_loss = 50
    start_time = time.time()
    for i in range(1, h_['epochs'] + 1):
        s_t = time.time()
        mean_loss = train_epoch_channels(dataloader=dataloader, 
                                         unet=unet, 
                                         diffused_model=diffused_model, 
                                         optimizer=optimizer, 
                                         scheduler=scheduler,
                                         device=device, 
                                         condition=meta['condition'], 
                                         number_of_repetition=1)
        logger.info(f'Epoch: {i}, mean loss: {mean_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}')
        if (mean_loss < min_loss):
            min_loss = mean_loss
            torch.save(unet.state_dict(), os.path.join(save_weights_path, 'unet_best.pth'))
            logger.info(f'epoch {i} unet_best.pth has been saved.')
        if (i % 50 == 0):
            torch.save(unet.state_dict(), os.path.join(save_weights_path, f'unet_{i}.pth'))

        e_t = time.time()
        logger.info(f"Epoch Time Used: {e_t - s_t}s; Total Time Used: {e_t - start_time}s")
