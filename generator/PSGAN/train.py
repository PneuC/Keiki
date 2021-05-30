import sys
sys.path.append('../..')
import os
import json
import time
import argparse
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from generator.dataset import AugmentedDataset
from generator.PSGAN.models import get_models
from root import PRJROOT


# Reference to https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?highlight=dcgan
def makedires(*paths):
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def sample_noise(n, zlen, device):
    zg = torch.rand(n, args.nzg, 1, device=device).expand(-1, -1, zlen)
    zl = torch.rand(n, args.nzl, zlen, device=device) if args.nzl > 0 else None
    return zg, zl


class MeanVarLoss(nn.Module):
    def __init__(self):
        super(MeanVarLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x1, x2):
        v1, m1 = torch.var_mean(x1, dim=0)
        v2, m2 = torch.var_mean(x2, dim=0)
        return self.mse(v1, v2) + self.mse(m1, m2)


def train(dataset, G, D, device):

    start_time = time.time()
    loss_func = nn.BCELoss()
    optG = Adam(G.parameters(), lr=args.lr)
    optD = Adam(D.parameters(), lr=args.lr)
    G_losses = []
    D_losses = []
    print('Traning Start')
    for t in range(args.n_epochs):
        G_losses.append(0.0)
        D_losses.append(0.0)
        total = 0
        for s in range(0, len(dataset), args.batch_size):
            n = min(args.batch_size, len(dataset) - s)
            real = dataset[s: s + n].to(device)
            # Train Discriminator to maximize real samples
            # -------------------
            D.zero_grad()
            y_real = D(real)
            loss_real = loss_func(y_real, torch.ones_like(y_real))
            D_losses[-1] += n * loss_real.item()
            loss_real.backward()

            # Train Discriminator to minimize fake samples
            # -------------------
            zg, zl = sample_noise(n, args.lz, device)
            fake = G(zg, zl, device=device)
            y_fake = D(fake)
            loss_fake = loss_func(y_fake, torch.zeros_like(y_fake))
            D_losses[-1] += n * loss_fake.item()
            loss_fake.backward()
            optD.step()

            # Train Generator to maximize fake samples
            # ---------------
            for _ in range(args.g_credits):
                G.zero_grad()
                zg, zl =sample_noise(n, args.lz, device)
                fake = G(zg, zl, device=device)
                y_fake = D(fake)
                loss_g = loss_func(y_fake, torch.ones_like(y_fake))
                G_losses[-1] += loss_g.item() * n
                loss_g.backward()
                optG.step()
            total += n
            # Matching mean and variance loss to overcome mode collapse
        G_losses[-1] /= (total * args.g_credits)
        D_losses[-1] /= (total * 2)
        if t % 10 == 9:
            print('Epoch %d, G loss=%.6f, D loss=%.6f, time %.2fs'
                  % (t+1, G_losses[-1], D_losses[-1], time.time()-start_time))

        if t % args.save_interval == args.save_interval-1:
            print('saving model and samples ... ', end='')
            torch.save(G.state_dict(), args.result_path + "/model/DCGAN_%d.pkl" % (t + 1))
            zg, zl = sample_noise(30, args.lz, device)
            with torch.no_grad():
                G.eval()
                samples = G(zg, zl, device=device).to('cpu').numpy()
                G.train()
            directory = args.result_path + '/samples/iteration%d' % (t + 1)
            makedires(directory)
            for i, item in enumerate(samples):
                item = item.transpose()
                np.save(directory + '/sample_%d' % (i + 1), item)
            print('finish')

    with open(args.result_path + '/G_loss.json', 'w') as f:
        json.dump(G_losses, f)
    with open(args.result_path + '/D_loss.json', 'w') as f:
        json.dump(D_losses, f)


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:%d" % args.gpuid if (torch.cuda.is_available() and args.gpuid >= 0) else "cpu")
    G, D = get_models(args.nzg, args.nzp, args.min_channels, device)
    dataset = AugmentedDataset(args.data_path, datalen=64, transpose=True)
    print(G, '\n', D)
    makedires(args.result_path  + '/samples', args.result_path  + '/model')
    train(dataset, G, D, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=PRJROOT + 'data/code')
    parser.add_argument('--n_epochs', type=int, default=5000,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='save every SAVE_INTERVAL epochs')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='ID of gpu to use, if negative or not valide than use cpu')

    parser.add_argument('--g_credits', type=int, default=2, help='')
    parser.add_argument('--nzg', type=int, default=64, help='channels of the global noise vector')
    parser.add_argument('--nzp', type=int, default=16, help='channels of the local noise vector')
    parser.add_argument('--nzl', type=int, default=0, help='channels of the noise vector')
    parser.add_argument('--lz', type=int, default=10, help='length of the noise vector')
    parser.add_argument('--min_channels', type=int, default=64)
    parser.add_argument('--result_path', type=str, default='./result')
    args = parser.parse_args()
    main()
