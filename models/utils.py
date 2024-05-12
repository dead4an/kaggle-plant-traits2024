# Imports
import os

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm


# Torch options
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Datasets
class TrainDataset(Dataset):
    def __init__(self, df: np.ndarray, img_folder: str) -> None:
        self.df = df
        self.img_folder = img_folder
        self.transforms = v2.Compose([
            v2.RandomPerspective(distortion_scale=0.5, p=0.5),
            v2.RandomRotation(degrees=(0, 180)),
            v2.RandomHorizontalFlip(p=0.5)
        ])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        # Get data from dataframe
        img_id = self.df[idx, 0].astype(int)
        features = self.df[idx, 1:164]
        targets = self.df[idx, 164:170]

        features = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        targets = torch.tensor(targets, dtype=torch.float32).to(DEVICE)

        # Get image
        img_path = os.path.join(self.img_folder, f'{img_id}.jpeg')
        img = cv2.imread(img_path)
        img = torch.tensor(img, dtype=torch.float32).to(DEVICE)
        img = img.permute(2, 0, 1)
        img = self.transforms(img)
        img /= 255.0
        
        return img, features, targets

    def __len__(self):
        return len(self.df)
    

class TestDataset(Dataset):
    def __init__(self, df: np.ndarray, img_folder: str) -> None:
        self.df = df
        self.img_folder = img_folder
    
    def __getitem__(self, idx):
        # Get data from dataframe
        img_id = self.df[idx, 0].astype(int)
        features = self.df[idx, 1:164]
        
        features = torch.tensor(features, dtype=torch.float32).to(DEVICE)

        # Get image
        img_path = os.path.join(self.img_folder, f'{img_id}.jpeg')
        img = cv2.imread(img_path)
        img = torch.tensor(img, dtype=torch.float32).to(DEVICE)
        img = img.permute(2, 0, 1)
        img /= 255.0

        return img, features
    
    def __len__(self):
        return len(self.df)
    

# Model usage
def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer,
                criterion: callable, epoch: int=0) -> None:
    model.train()
    total_loss = 0
    total_r2 = 0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        images, features, targets = batch
        output = model(images, features)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss
        total_r2 += r2_score(output, targets)

    total_loss /= len(dataloader)
    total_r2 /= len(dataloader)
    print(f'Epoch: {epoch} | Loss: {total_loss} | R2: {total_r2}')

def evaluate(model: nn.Module, dataloader: DataLoader, 
             criterion: callable) -> None:
    model.eval()
    total_loss = 0
    total_r2 = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, features, targets = batch
            output = model(images, features)
            loss = criterion(output, targets)
            total_loss += loss
            total_r2 += r2_score(output, targets)

    total_loss /= len(dataloader)
    total_r2 /= len(dataloader)
    print(f'Loss: {total_loss} | R2: {total_r2}')

def inference(model: nn.Module, dataloader: DataLoader) -> list:
    model.eval()
    output = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, features = batch
            batch_output = model(images, features)
            output.extend(batch_output.tolist())

    return output

def r2_score(y_predicted: torch.Tensor, y_true: torch.Tensor) -> torch.float32:
    return torch.sum((y_true - y_predicted)**2) \
        / torch.sum((y_true - y_true.mean())**2)

# Data management
def load_train_data(path: str) -> tuple[np.ndarray]:
    df = pd.read_csv(path)

    target_columns = ['X4_mean', 'X11_mean', 'X18_mean', 
                          'X26_mean', 'X50_mean', 'X3112_mean']
    targets = df[target_columns].to_numpy()
    features = df.iloc[:, :164].to_numpy()
    return features, targets

def preprocess_data(df: np.ndarray, scaler: any=None) -> np.ndarray:
    indexes = df[:, 0].reshape(-1, 1)
    features = df[:, 1:]
    features = scaler.transform(features)
    return np.hstack((indexes, features))

# Model checkpoints management
def save_checkpoint(model: nn.Module, optimizer: Optimizer, scaler: callable,
                    path: str, epoch: int=0) -> None:
    torch.save({
        'model_state': model.state_dict(),
        'optimizer': optimizer,
        'scaler': scaler,
        'epoch': epoch
    }, path)

def load_checkpoint(path: str) -> tuple:
    checkpoint = torch.load(path)
    model_state = checkpoint['model_state']
    optimizer = checkpoint['optimizer']
    scaler = checkpoint['scaler']
    epoch = checkpoint['epoch']

    return model_state, optimizer, scaler, epoch
