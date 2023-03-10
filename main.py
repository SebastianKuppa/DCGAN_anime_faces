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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data preprocessing
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
train_dataset = datasets.ImageFolder(root="anime", transform=train_transform,)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)


# weight initialization
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Block1
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Block2
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Block3
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Block4
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Block5
            nn.Conv2d(64 * 8, 1, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# init generator and discriminator instances and their weights
generator = Generator().to(device=device)
generator.apply(init_weights)
discriminator = Discriminator()
discriminator.apply(init_weights)

# init BCELoss
loss_function = nn.BCELoss()


def generator_loss(discriminator_prediction_label, ground_truth_label):
    return loss_function(discriminator_prediction_label, ground_truth_label)


def discriminator_loss(output, label):
    return loss_function(output, label)


# optimizer init
lr = 0.0002
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


#train loop
for epoch in range(1, num_epochs+1):
    d_loss_list, g_loss_list = [], []
    for index, (real_images, _) in enumerate(train_loader):
        # discriminator training
        d_optimizer.zero_grad()
        real_images = real_images.to(device)
        real_target = Variable(torch.ones(real_images.size(0)).to(device))
        fake_target = Variable(torch.zeros(real_images.size(0)).to(device))

        output = discriminator(real_images)
        d_real_loss = discriminator_loss(output, real_target)
        d_real_loss.backward()

        # init noise vector for generator input
        noise_vector = torch.randn(real_images.size(0), 100, 1, 1, device=device)
        generated_image = generator(noise_vector)
        output = discriminator(generated_image.detach())
        d_fake_loss = loss_function(output, fake_target)
        d_fake_loss.backward()

        # calc total loss
        d_total_loss = d_fake_loss + d_real_loss
        d_loss_list.append(d_total_loss)
        # update the discriminator parameters using the Adam optimizer
        d_optimizer.step()

        # generator training
        g_optimizer.zero_grad()
        generated_output = discriminator(generated_image)
        g_loss = generator_loss(generated_output, real_target)
        g_loss_list.append(g_loss)

        g_loss.backward()
        g_optimizer.step()
