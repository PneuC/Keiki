import torch
from torch import nn


def get_models(nzg=64, nzl=32, nzp=16, n_hidden=32, h_channels=24, n_stack=3, device='cpu'):
    generator = PeriodicGenerator(nzg, nzl, nzp, n_hidden, h_channels, n_stack).to(device)
    predictor = LSTM(h_channels, n_hidden, h_channels, n_stack).to(device)
    discriminator = LSTM(h_channels, n_hidden, 1, n_stack).to(device)
    embedder = LSTM(15, n_hidden, h_channels, n_stack).to(device)
    reconstructor = LSTM(h_channels, n_hidden, 15, n_stack).to(device)
    return generator, predictor, discriminator, embedder, reconstructor


class LSTM(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, n_stack):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(n_in, n_hidden, num_layers=n_stack, batch_first=True)
        self.n_hidden, self.n_stack = n_hidden, n_stack
        self.main = nn.Sequential(
            nn.Linear(n_hidden, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.main(lstm_out)


class MixedReconstructor(nn.Module):
    def __init__(self, n_hidden, h_channels, n_stack):
        super().__init__()
        self.lstm = nn.LSTM(h_channels, n_hidden, num_layers=n_stack, batch_first=True)
        self.btype_out = nn.Sequential(nn.Linear(n_hidden, 15), nn.Softmax(dim=2))
        self.color_out = nn.Sequential(nn.Linear(n_hidden, 6), nn.Softmax(dim=2))
        self.numberical_out = nn.Sequential(nn.Linear(n_hidden, 13), nn.Sigmoid())

    def forward(self, h):
        lstm_out, _ = self.lstm(h)
        btype = self.btype_out(lstm_out)
        color = self.color_out(lstm_out)
        numberical = self.numberical_out(lstm_out)
        return torch.cat([btype, color, numberical], dim=2)


class PeriodicGenerator(nn.Module):
    def __init__(self, nzg, nzl, nzp, n_hidden, n_out, n_stack):
        super(PeriodicGenerator, self).__init__()
        self.zp_mid = nn.Sequential(nn.Linear(nzg, nzp), nn.Sigmoid())
        self.k_gen = nn.Linear(nzp, nzp)
        self.phi_gen = nn.Linear(nzp, nzp)
        self.nzp = nzp
        self.lstm = LSTM(nzg + nzl + nzp, n_hidden, n_out, n_stack)

    def forward(self, zg, zl, device='cpu'):
        batch_size, seqlen, _ = zg.shape
        mid = self.zp_mid(zg)
        k = self.k_gen(mid)
        phi = self.phi_gen(mid)
        zp = torch.zeros(batch_size, seqlen, self.nzp, device=device)
        for t in range(seqlen):
            zp[:, t, :] = t
        zp = torch.sin(k * zp + phi)
        if not zl is None:
            z = torch.cat([zg, zl, zp], dim=2)
        else:
            z = torch.cat([zg, zp], dim=2)
        return self.lstm(z)
