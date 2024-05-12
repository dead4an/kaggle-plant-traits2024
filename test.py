# Imports
import os

import torch
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_m

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.resnet import MultiInputResNet
from models.linear import LinearNet
from models.utils import TestDataset, inference, load_checkpoint, preprocess_data


# Torch options
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Paths
ROOT = os.getcwd()
TEST_IMG_FOLDER = os.path.join(ROOT, 'test_images')
TEST_PATH = os.path.join(ROOT, 'test.csv')
CHECKPOINTS_FOLDER = os.path.join(ROOT, 'checkpoints')
NUM_TABULAR_FEATURES = 163
LINEAR_NET_OUTPUT_DIM = 128

# Load test dataset
test_df = pd.read_csv(TEST_PATH).to_numpy()

# Load checkpoint
checkpoint_path = os.path.join(CHECKPOINTS_FOLDER, 'efficientnetv2_m_5.pt')
checkpoint = torch.load(checkpoint_path)

backbone = efficientnet_v2_m()
linear = LinearNet(NUM_TABULAR_FEATURES, LINEAR_NET_OUTPUT_DIM)
model = MultiInputResNet(backbone, linear, LINEAR_NET_OUTPUT_DIM, 6).to(DEVICE)
model.load_state_dict(checkpoint['model_state'])
scaler = checkpoint['scaler']

# Preprocess data
test_data = preprocess_data(test_df, scaler)

# Create test dataset and dataloader
test_dataset = TestDataset(test_data, TEST_IMG_FOLDER)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=16
)

# Run inference
if __name__ == '__main__':
    output = inference(model, test_dataloader)
    target_columns = ['X4', 'X11', 'X18', 'X26', 'X50', 'X3112']
    submit_df = pd.DataFrame(data=output, columns=target_columns)
    submit_df.index = test_data[:, 0].astype(int)
    submit_df.index.name = 'id'
    submit_df.to_csv('submit.csv')
