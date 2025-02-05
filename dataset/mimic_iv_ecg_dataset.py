import os
import torch
from torch.utils.data import Dataset

class DictDataset(Dataset):
    def __init__(self, path:str):
        self.data_dict = torch.load(path)
        self.keys = list(self.data_dict.keys()) 

    def __len__(self):
        return len(self.data_dict) 
    
    def __getitem__(self, idx):
        key = self.keys[idx] 
        latent_dict = self.data_dict[key] 

        return latent_dict['data'], latent_dict['label']


class VAE_MIMIC_IV_ECG_Dataset(Dataset):
    def __init__(self, path:str, usage='all'):
        self.path = path
        self.file_list = os.listdir(path)
        # Make sure every time the order of list is the same
        # so as the train and test fold if split in training
        self.file_list.sort(key=lambda x: int(x.split('.')[0]))

        if usage == 'test':
            self.file_list = self.file_list[:50000]

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index) -> tuple:
        latent_file = self.file_list[index]
        latent_dict = torch.load(os.path.join(self.path, latent_file), map_location='cpu')

        # data: (C, L) i.e. (4, 128)
        # label: dict contain keys of [text, subject_id, age, gender]
        return (latent_dict['data'], latent_dict['label'])
