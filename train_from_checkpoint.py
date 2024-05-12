# Imports
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from utils import TrainDataset, train_epoch, evaluate, preprocess_data


# Torch options
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Paths
ROOT = os.getcwd()
TRAIN_IMG_FOLDER = os.path.join(ROOT, 'train_images')
TRAIN_PATH = os.path.join(ROOT, 'train.csv')
CHECKPOINTS_FOLDER = os.path.join(ROOT, 'checkpoints')

# Load train dataset
train_df = pd.read_csv(TRAIN_PATH)

# Train/val split
target_columns = ['X4_mean', 'X11_mean', 'X18_mean', 
                  'X26_mean', 'X50_mean', 'X3112_mean']
features = train_df.iloc[:, :164].to_numpy()
targets = train_df[target_columns].to_numpy()
X_train, X_val, y_train, y_val = train_test_split(
    features, targets, test_size=0.2
)

# Parameters
START_EPOCH = 5
NUM_EPOCHS = 100
LR = 6e-3
SAVE_CHECKPOINTS = True

# Load checpoint
checkpoint_path = os.path.join(CHECKPOINTS_FOLDER, f'resnet50_{START_EPOCH}.pt')
checkpoint = torch.load(checkpoint_path)

# Preprocess data
scaler = checkpoint['scaler']
train_data = preprocess_data(X_train, scaler)
val_data = preprocess_data(X_val, scaler)

# Concat
train_data = np.hstack((X_train, y_train))
val_data = np.hstack((X_val, y_val))

# Create train/val dataloaders
train_dataset = TrainDataset(train_data, TRAIN_IMG_FOLDER)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    drop_last=True
)

val_dataset = TrainDataset(val_data, TRAIN_IMG_FOLDER)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=32
)

# Model and optimization
model = checkpoint['model']
optimizer = optim.Adam(model.parameters(), LR)
criterion = nn.SmoothL1Loss().to(DEVICE)

# Training
for epoch in range(START_EPOCH + 1, NUM_EPOCHS + START_EPOCH):
    train_epoch(model, train_dataloader, optimizer, criterion, epoch)
    evaluate(model, val_dataloader, criterion)
    if epoch % 5 == 0 and SAVE_CHECKPOINTS:
        checkpoint_path = os.path.join(CHECKPOINTS_FOLDER, f'resnet50_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'scaler': scaler
        }, checkpoint_path)
