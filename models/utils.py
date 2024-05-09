# Imports
import os

import torch
from torch.utils.data import Dataset

import cv2
import pandas as pd


# Torch options
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Datasets
class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_folder: str) -> None:
        self.df = df
        self.img_folder = img_folder

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        # Get data from dataframe
        features = self.df.iloc[idx, :164]
        targets = self.df.iloc[idx, 164:170]

        features = torch.tensor(features).to(DEVICE)
        targets = torch.tensor(targets).to(DEVICE)

        # Get image
        img_path = os.path.join(self.img_folder, f'{features.name}.jpeg')
        img = cv2.imread(img_path)
        img = torch.tensor(img).to(DEVICE)
        img /= 255
        
        return img, features, targets

    def __len__(self):
        return len(self.df)
