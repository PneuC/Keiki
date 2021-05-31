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
from generator.dataset import  AugmentedDataset
from generator.DCGAN.models import get_models
from root import PRJROOT


def makedires(*paths):
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def train(dataset, G, D, device):
    start_time = time.time()
    loss_func = nn.BCELoss()
    optG = Adam(G.parameters(), lr=args.lr)
    optD = Adam(D.parameters(), lr=args.lr)
    G_losses = []
    D_losses = []
    MV_loss = []

    print('Traning Start')
    for t in range(args.n_epochs):
        G_losses.append(0.0)
        D_losses.append(0.0)
        MV_loss.append(0.0)
        total = 0
        # dataset.shuffle()
        for s in range(0, len(dataset), args.batch_size):
            n = min(args.batch_size, len(dataset) - s)
            real = dataset[s: s + n].to(device)
            # Train Discriminator to maximize real samples
            # -------------------
            for _ in range(args.d_credits):
                D.zero_grad()
                y_real = D(real)
                loss_real = loss_func(y_real, torch.ones_like(y_real))
                D_losses[-1] += n * loss_real.item()
                loss_real.backward()

                # Train Discriminator to minimize fake samples
                # -------------------
                z = torch.randn(n, args.nz, 1, device=device)
                fake = G(z)
                y_fake = D(fake)
                loss_fake = loss_func(y_fake, torch.zeros_like(y_fake))
                D_losses[-1] += n * loss_fake.item()
                loss_fake.backward()
                optD.step()

            # Train Generator to maximize fake samples
            # ---------------
            for _ in range(args.g_credits):
                G.zero_grad()
                z = torch.randn(n, args.nz, 1, device=device)
                fake = G(z)
                y_fake = D(fake)
                loss_g = loss_func(y_fake, torch.ones_like(y_fake))
                G_losses[-1] += loss_g.item() * n
                loss_g.backward()
                optG.step()
            total += n
        G_losses[-1] /= (total * args.g_credits)
        D_losses[-1] /= (total * 2 * args.d_credits)

        if t % 10 == 9:
            print('Epoch %d, G loss=%.6f, D loss=%.6f, time %.2fs'
                  % (t+1, G_losses[-1], D_losses[-1], time.time() - start_time))

        if t % args.save_interval == args.save_interval-1:
            with torch.no_grad():
                G.eval()
                z = torch.randn(50, args.nz, 1, device=device)
                samples = G(z).to('cpu').numpy()
                G.train()
            directory = args.result_path + '/samples/iteration%d' % (t + 1)
            makedires(directory)
            for i, item in enumerate(samples):
                item = item.transpose()
                np.save(directory + '/sample_%d' % (i + 1), item)
        if t % 50 == 49:
            torch.save(G.state_dict(), args.result_path + "/model/DCGAN_%d.pkl" % (t + 1))

    with open(args.result_path + '/G_loss.json', 'w') as f:
        json.dump(G_losses, f)
    with open(args.result_path + '/D_loss.json', 'w') as f:
        json.dump(D_losses, f)


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:%d" % args.gpuid if (torch.cuda.is_available() and args.gpuid >= 0) else "cpu")
    G, D = get_models(args.nz, args.min_channels, device)
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

    parser.add_argument('--g_credits', type=int, default=10, help='')
    parser.add_argument('--d_credits', type=int, default=1, help='')
    parser.add_argument('--nz', type=int, default=64, help='channels of the global noise vector')
    parser.add_argument('--min_channels', type=int, default=32)
    parser.add_argument('--result_path', type=str, default='./result')
    args = parser.parse_args()
    main()
