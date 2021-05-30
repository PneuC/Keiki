import torch
from torch import nn


def get_models(nz, min_layers, device):
    generator = Generator(nz, min_layers).to(device)
    discrimintor = Discriminator(min_layers).to(device)
    return generator, discrimintor


class Generator(nn.Module):
    def __init__(self, nz=64, ncf=32):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(nz, ncf << 3, 4, 1, 0), nn.BatchNorm1d(ncf << 3), nn.ReLU(),      # 4
            nn.ConvTranspose1d(ncf << 3, ncf << 2, 4, 2, 1), nn.BatchNorm1d(ncf << 2), nn.ReLU(),# 8
            nn.ConvTranspose1d(ncf << 2, ncf << 1, 4, 2, 1), nn.BatchNorm1d(ncf << 1), nn.ReLU(),# 16
            nn.ConvTranspose1d(ncf << 1, ncf, 4, 2, 1), nn.BatchNorm1d(ncf), nn.ReLU(),# 32
            nn.ConvTranspose1d(ncf, 15, 4, 2, 1), nn.Sigmoid(),          # 64
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, nci=32):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(15, nci, 4, 2, 1), nn.BatchNorm1d(nci), nn.ReLU(),         # 32
            nn.Conv1d(nci, nci << 1, 4, 2, 1), nn.BatchNorm1d(nci << 1), nn.ReLU(),         # 16
            nn.Conv1d(nci << 1, nci << 2, 4, 2, 1), nn.BatchNorm1d(nci << 2), nn.ReLU(),         # 8
            nn.Conv1d(nci << 2, nci << 3, 4, 2, 1), nn.BatchNorm1d(nci << 3), nn.ReLU(),         # 4
            nn.Conv1d(nci << 3, 1, 4, 1, 0), nn.Sigmoid()   # 1
        )

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    Zg = torch.rand(2, 10, 1).expand(-1, -1, 10)
