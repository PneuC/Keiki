import sys
sys.path.append('../..')
import json
import os
import time
import argparse
import torch
import numpy as np
from torch import nn
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from generator.dataset import AugmentedDataset
from generator.TimeGAN.models import get_models
from root import PRJROOT


class MeanVarLoss(nn.Module):
    def __init__(self):
        super(MeanVarLoss, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, x1, x2):
        v1, m1 = torch.var_mean(x1, dim=0)
        v2, m2 = torch.var_mean(x2, dim=0)
        return self.mae(torch.sqrt(v1), torch.sqrt(v2)) + self.mae(m1, m2)


def makedires(*paths):
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

def sampling(G, P, R, seqlen, n=1, device='cpu'):
    zg = torch.rand(n, 1, args.nzg, device=device).expand((-1, seqlen, -1))
    zl = torch.rand(n, seqlen, args.nzl, device=device) if args.nzl else None
    with torch.no_grad():
        data = R(P(G(zg, zl, device=device))).to('cpu')
    return data.numpy()

def train(dataset, models, device):
    print('Training start')
    G, P, D, E, R = models
    MSE = nn.MSELoss()
    BCE = nn.BCELoss()
    MV = MeanVarLoss()
    optG = RMSprop(G.parameters(), lr=args.lr)
    optP = RMSprop(P.parameters(), lr=args.lr)
    optD = RMSprop(D.parameters(), lr=args.lr, weight_decay=1e-4)
    optE = RMSprop(E.parameters(), lr=args.lr)
    optR = RMSprop(R.parameters(), lr=args.lr)
    ################################################
    # Pretrain emebedder & reconstructor
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size // 2, shuffle=True)
    start_time = time.time()
    for t in range(args.r_epoches):
        loss_sum = 0.0
        total = 0
        for i, x in enumerate(dataloader):
            x = x.to(device)
            total += x.shape[0]
            E.zero_grad()
            R.zero_grad()
            x_hat = R(E(x))
            r_loss = MSE(x, x_hat)
            loss_sum += r_loss.item() * x.shape[0]
            r_loss.backward()
            optE.step()
            optR.step()
        # Checkpoints
        if t % 10 == 9:
            stage_time = time.time() - start_time
            print('Pretraining E & R, iteration %d / %d, reconstruction loss: %.6f, time: %.1fs'
                  % (t + 1, args.r_epoches, loss_sum / total, stage_time))
    ################################################
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    stage1_cost = time.time() - start_time
    ################################################
    # Pretrain predictor with supervised loss only
    stage_start = time.time()
    for t in range(args.s_epoches):
        loss_sum = 0.0
        total = 0
        for i, x in enumerate(dataloader):
            x = x.to(device)
            n = x.shape[0]
            P.zero_grad()
            with torch.no_grad():
                h_real = E(x)
            h_fake = P(h_real)
            s_loss = MSE(h_real[:, 1:, :], h_fake[:, :-1, :])
            loss_sum += s_loss.item() * n
            s_loss.backward()
            optP.step()
            total += n
        # Checkpoints
        if t % 10 == 9:
            stage_time = time.time() - stage_start
            total_time = time.time() - start_time
            print('Pretraining P, iteration %d / %d, supervised loss: %.6f, stage time: %.1fs, total time: %.1fs'
                  % (t + 1, args.s_epoches, loss_sum / total,  stage_time, total_time))
    ################################################
    stage2_cost = time.time() - stage_start
    ################################################
    # Jointly training
    stage_start = time.time()
    loss_record = []

    for t in range(args.j_epoches):
        total = 0
        loss_record.append({'R-loss': 0., 'G-loss': 0., 'G-loss-u': 0., 'G-loss-s': 0., 'G-loss-v': 0., 'D-loss': 0.})
        for i, x in enumerate(dataloader):
            n = x.shape[0]
            x = x.to(device)
            # Train D
            for _ in range(args.d_credits):
                D.zero_grad()
                zg = torch.rand(n, 1, args.nzg, device=device).expand((-1, args.seqlen, -1))
                zl = torch.rand(n, args.seqlen, args.nzl, device=device) if args.nzl else None
                h_real = E(x)
                h_fake0 = G(zg, zl, device=device)
                h_fake1 = P(h_fake0)
                y_real = D(h_real)
                y_fake0 = D(h_fake0)
                y_fake1 = D(h_fake1)
                loss_real = BCE(y_real, torch.ones(n, args.seqlen, 1, device=device))
                loss_fake0 = BCE(y_fake0, torch.zeros(n, args.seqlen, 1, device=device))
                loss_fake1 = BCE(y_fake1, torch.zeros(n, args.seqlen, 1, device=device))
                batch_loss = loss_real.item() + loss_fake0.item() + loss_fake1.item()
                loss_record[-1]['D-loss'] += batch_loss * n / 3 / args.d_credits
                optD.step()
            # Train E, R
            E.zero_grad()
            R.zero_grad()
            h_real = E(x)
            x_tilde = R(h_real)
            r_loss = MSE(x, x_tilde)
            loss_record[-1]['R-loss'] += r_loss.item() * n
            (r_loss * 10).backward()
            h_real = E(x)
            h_fake = P(h_real)
            s_loss = MSE(h_real[:, 1:, :], h_fake[:, :-1, :])
            (s_loss * 0.1).backward()
            optE.step()
            optR.step()
            # Train G, P
            for _ in range(args.g_credits):
                G.zero_grad()
                P.zero_grad()
                h_real = E(x)
                h_fake = P(h_real)
                # Supervised Loss
                s_loss = MSE(h_real[:, 1:, :], h_fake[:, :-1, :]) * 10
                loss_record[-1]['G-loss-s'] += s_loss.item() * n / args.g_credits
                (s_loss * 100).backward()
                # Adversarial Loss
                zg = torch.rand(n, 1, args.nzg, device=device).expand((-1, args.seqlen, -1))
                zl = torch.rand(n, args.seqlen, args.nzl, device=device) if args.nzl else None
                h_fake0 = G(zg, zl, device=device)
                h_fake1 = P(h_fake0)
                y_fake0 = D(h_fake0)
                y_fake1 = D(h_fake1)
                u_loss = BCE(y_fake0, torch.ones(n, args.seqlen, 1, device=device)) \
                         + BCE(y_fake1, torch.ones(n, args.seqlen, 1, device=device))
                loss_record[-1]['G-loss-u'] += u_loss.item() * n / args.g_credits / 2
                u_loss.backward()
                # Mean-Variance Loss
                zg = torch.rand(n, 1, args.nzg, device=device).expand((-1, args.seqlen, -1))
                zl = torch.rand(n, args.seqlen, args.nzl, device=device) if args.nzl else None
                h_real = E(x)
                h_fake = P(G(zg, zl, device=device))
                mv_loss = MV(h_real, h_fake)
                loss_record[-1]['G-loss-v'] += mv_loss.item() * n / args.g_credits
                (mv_loss * 100).backward()
                optP.step()
                optG.step()
            total += n

        for key in loss_record[-1].keys():
            loss_record[-1][key] = loss_record[-1][key] / total
        # Checkpoints
        if t % 10 == 9:
            stage_time = time.time() - stage_start
            total_time = time.time() - start_time
            print(
                'Jointly training, iteration %d / %d, Reconstruction loss: %.6f,  G adversarial-loss: %.6f, G supervi'
                'sed-loss: %.6f, G mean-variance-loss: %.6f, D loss: %.6f, stage time: %.1fs, total time: %.1fs' % (
                    t + 1, args.j_epoches, loss_record[-1]['R-loss'], loss_record[-1]['G-loss-u'],
                    loss_record[-1]['G-loss-s'], loss_record[-1]['G-loss-v'], loss_record[-1]['D-loss'],
                    stage_time, total_time
                )
            )
        if t % args.save_interval == args.save_interval - 1:
            model_path = args.result_path + '/model/iteration%d' % (t+1)
            makedires(model_path)
            torch.save(R, model_path + '/Reconstructor.pkl')
            torch.save(G, model_path + '/Generator.pkl')
            torch.save(P, model_path + '/Predictor.pkl')

            with open('loss.json', 'w') as f:
                json.dump(loss_record, f)
            samples = sampling(G, P, R, args.seqlen, 50, device)

            directory = args.result_path + '/joint train samples/iteration%d' % (t + 1)
            makedires(directory)
            for i, item in enumerate(samples):
                np.save(directory + '/original_%d' % (i + 1), item)

    ################################################
    stage3_cost = time.time() - stage_start
    total_time = time.time() - start_time
    print('Training finish, stage time costs: %.1fs, %.1fs, %.1fs, total time: %.1fs'
          % (stage1_cost, stage2_cost, stage3_cost, total_time))


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:%d" % args.gpu if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")
    print(device)
    dataset = AugmentedDataset(args.data_path, args.seqlen)
    print(len(dataset))
    print('result path: ' + args.result_path)
    models = get_models(args.nzg, args.nzl, args.nzp, args.nhidden, args.h_channels, args.nstack, device)
    makedires(args.result_path)
    train(dataset,  models, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=PRJROOT + 'data/code')
    parser.add_argument('--r_epoches', type=int, default=5000, help='number of epochs of pretraining emebber and reconstructor')
    parser.add_argument('--s_epoches', type=int, default=500, help='number of epochs of pretraining predictor')
    parser.add_argument('--j_epoches', type=int, default=5000, help='number of epochs of joint training')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--save_interval', type=int, default=100, help='save every SAVE_INTERVAL epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id. If negative use CPU')
    parser.add_argument('--nzg', type=int, default=64, help='channels of the global noise vector')
    parser.add_argument('--nzp', type=int, default=16, help='channels of the periodic noise vector')
    parser.add_argument('--nzl', type=int, default=0, help='channels of the local noise vector')
    parser.add_argument('--seqlen', type=int, default=64, help='length of the sequence')
    parser.add_argument('--d_credits', type=int, default=1, help='update discriminator how many times in each epoch')
    parser.add_argument('--g_credits', type=int, default=1, help='update generator how many times in each epoch')
    parser.add_argument('--nhidden', type=int, default=128, help='hidden size for all')
    parser.add_argument('--h_channels', type=int, default=24, help='number of dimensions of hidden code')
    parser.add_argument('--nstack', type=int, default=3, help='number of stacked layers')
    parser.add_argument('--result_path', type=str, default='./result')
    args = parser.parse_args()
    main()



