import numpy as np
import torch
from torch.utils.data import Dataset
from data_make import get_classes, DanmakuEncoder
from random import shuffle


class DanmakuDataset(Dataset):
    def __init__(self, path, datalen=200, transpose=False, device='cpu'):
        super(DanmakuDataset, self).__init__()
        self.data = []
        fnum = 0
        while True:
            try:
                fpath = path + '/D%d.npy' % fnum
                original = np.load(fpath)
                if not transpose:
                    self.data.append(torch.tensor(original[:datalen], device=device).double())
                else:
                    npdata = original[:datalen].transpose()
                    self.data.append(torch.tensor(npdata, device=device).double())
                fnum += 1
            except FileNotFoundError:
                break

    def __getattr__(self, item):
        if item == 'm':
            return self.data[0].size()[0]
        if item == 'n_fea':
            return 15

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        shuffle(self.data)

    def __getitem__(self, item):
        if type(item) == int:
            return self.data[item]
        else:
            return torch.stack(self.data[item], dim=0)


class AugmentedDataset(Dataset):
    def __init__(self, path, datalen=100, one_hot=False, transpose=False):
        print(path)
        self.danmakus = get_classes(path)
        self.datalen = datalen
        self.one_hot = one_hot
        self.transpose = transpose
        DanmakuEncoder()

    def __len__(self):
        return len(self.danmakus)

    def __getitem__(self, item):
        cls = self.danmakus[item]
        if type(item) == int:
            oridata = DanmakuEncoder.inst().class_to_ps(cls, L=self.datalen, augmentation=True, onehot=self.one_hot)
            if self.transpose:
                oridata = oridata.transpose()
            return torch.tensor(oridata).double()
        else:
            oridata = []
            for danmaku in cls:
                param_seq = DanmakuEncoder.inst().class_to_ps(danmaku, L=self.datalen, augmentation=True, onehot=self.one_hot)
                if self.transpose:
                    param_seq = param_seq.transpose()
                oridata.append(torch.tensor(param_seq).double())
            return torch.stack(oridata, dim=0)

