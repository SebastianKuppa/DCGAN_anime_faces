import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np

batch_size = 32

# data preprocessing
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

train_dataset = datasets.ImageFolder(root="anime", transform=train_transform,)
train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True,)
