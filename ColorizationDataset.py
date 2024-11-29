import torch
import os
from torch.utils.data import Dataset

class LABDataset(Dataset):

    def __init__(self, l_path, ch_path):  
        self.l_path = l_path
        self.ch_path = ch_path  
        self.l_files = sorted([f for f in os.listdir(l_path) if f.endswith('.pt')])
        self.ch_files = sorted([f for f in os.listdir(ch_path) if f.endswith('.pt')])

    def __len__(self):
        return len(self.l_files)

    def __getitem__(self, idx):
        L_tensor = torch.load(os.path.join(self.l_path, self.l_files[idx]), weights_only=False)
        CH_tensor = torch.load(os.path.join(self.ch_path, self.ch_files[idx]), weights_only=False)
        return L_tensor, CH_tensor
