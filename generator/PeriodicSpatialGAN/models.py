import torch
from torch import nn


def get_models(nzg, nzp, min_layers, device):
    generator = Generator(nzg, nzp, min_layers).to(device)
    discrimintor = Discriminator(min_layers).to(device)
    return generator, discrimintor


class Generator(nn.Module):
    def __init__(self, nzg, nzp, ncf=64):
        super(Generator, self).__init__()
        self.zp_mid = nn.Sequential(nn.Conv1d(nzg, nzg, 1, 1), nn.ReLU())
        self.k_gen = nn.Conv1d(nzg, nzp, 1, 1)
        self.phi_gen = nn.Conv1d(nzg, nzp, 1, 1)
        self.nzp = nzp
        self.convs = nn.Sequential(
            nn.ConvTranspose1d(nzg + nzp, ncf << 2, 4, 1), nn.ReLU(),
            nn.ConvTranspose1d(ncf << 2, ncf << 1, 4, 2), nn.ReLU(),
            nn.ConvTranspose1d(ncf << 1, ncf, 4, 1), nn.ReLU(),
            nn.ConvTranspose1d(ncf, 15, 4, 2), nn.Sigmoid(),
        )

    def forward(self, zg, zl=None, device='cpu'):
        batch_size, _, length = zg.shape
        mid = self.zp_mid(zg)
        k = self.k_gen(mid)
        phi = self.phi_gen(mid)
        time = torch.zeros(batch_size, self.nzp, length, device=device)
        for t in range(length):
            time[:, :, t] = t
        zp = torch.sin(k * time + phi)
        if zl is None:
            z = torch.cat([zg, zp], dim=1)
        else:
            z = torch.cat([zg, zp, zl], dim=1)
        return self.convs(z)


class Discriminator(nn.Module):
    def __init__(self, ncf=32):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(15, ncf << 2, 4, 1), nn.ReLU(),
            nn.Conv1d(ncf << 2, ncf << 1, 4, 2), nn.ReLU(),
            nn.Conv1d(ncf << 1, ncf, 4, 1), nn.ReLU(),
            nn.Conv1d(ncf, 1, 4, 2), nn.Sigmoid(),
        )

    def forward(self, X):
        return self.main(X)

if __name__ == '__main__':
    G = Generator(16, 10)
    zg = torch.rand(1, 16, 10)
    print(G(zg).shape)
