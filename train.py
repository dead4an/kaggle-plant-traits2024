# Imports
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

from models.resnet import MultiInputResNet
from models.linear import LinearNet
from models.utils import TrainDataset, train_epoch, evaluate, preprocess_data, \
    load_train_data, save_checkpoint


# Torch options
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Paths
ROOT = os.getcwd()
TRAIN_IMG_FOLDER = os.path.join(ROOT, 'train_images')
TRAIN_PATH = os.path.join(ROOT, 'train_preprocessed.csv')
CHECKPOINTS_FOLDER = os.path.join(ROOT, 'checkpoints')

# Load train dataset
features, targets = load_train_data(TRAIN_PATH)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    features, targets, test_size=0.1
)

# Preprocess data
scaler = MinMaxScaler((-1, 1))
scaler.fit(X_train[:, 1:])
X_train = preprocess_data(X_train, scaler)
X_test = preprocess_data(X_val, scaler)

# Concat
train_data = np.hstack((X_train, y_train))
val_data = np.hstack((X_val, y_val))

# Create train/val dataloaders
train_dataset = TrainDataset(train_data, TRAIN_IMG_FOLDER)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=6,
    drop_last=True,
    shuffle=True
)

val_dataset = TrainDataset(val_data, TRAIN_IMG_FOLDER)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=10
)

# Parameters
NUM_EPOCHS = 100
LR = 4e-3
LINEAR_NET_OUTPUT_DIM = 128
NUM_TABULAR_FEATURES = 163
SAVE_CHECKPOINTS = True

# Model and optimization
backbone = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
linear = LinearNet(NUM_TABULAR_FEATURES, LINEAR_NET_OUTPUT_DIM)
model = MultiInputResNet(backbone, linear, LINEAR_NET_OUTPUT_DIM, 6).to(DEVICE)
optimizer = optim.Adam(model.parameters(), LR)
criterion = nn.SmoothL1Loss().to(DEVICE)

# Training
for epoch in range(NUM_EPOCHS):
    train_loss, train_r2 = train_epoch(model, train_dataloader, optimizer, 
                                       criterion, epoch)
    val_loss, val_r2 = evaluate(model, val_dataloader, criterion)
    if epoch % 5 == 0 and SAVE_CHECKPOINTS:
        checkpoint_path = os.path.join(CHECKPOINTS_FOLDER, 
                                       f'efficientnetv2_m_{epoch}.pt')
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            path=checkpoint_path,
            epoch=epoch,
            train_loss={'l1_loss': train_loss, 'r2_loss': train_r2},
            val_loss={'l1_loss': val_loss, 'r2_loss': val_r2}
        )