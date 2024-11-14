from torch.utils.data import Dataset, DataLoader
from augmentations import get_loading_transform_f16
import pandas as pd
from monai import data
import torch
import numpy as np





class HeadDataset(Dataset):
    def __init__(self, csv_file, data_augmentation, cache_dir="/scratch/ml9715/DINO-3D/temp_cache_large_3_channel/"):
        self.data = pd.read_csv(csv_file)
        # self.load = get_loading_transform_f16()
        self.load = get_loading_transform_f16()

        self.cache_dir = cache_dir
        self.cache_dataset = data.PersistentDataset(
            data=list([{"image":d} for d in self.data['img_path_T1_mni152']]), 
            transform=self.load, 
            cache_dir=self.cache_dir,
        )
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, 'img_path_T1_mni152']
        # image = self.load(img_path)
        try:
            image = self.cache_dataset.__getitem__(idx)['image']
            if image.shape[-1] <= 50:
                print("Error: {}".format(idx))
        except:
            image = torch.rand(1, 96, 96, 96)
            print("Created random image: {}".format(idx))
        if self.data_augmentation:
            image = self.data_augmentation(image)
        return image
    
    









