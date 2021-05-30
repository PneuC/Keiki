import os
import inspect
import importlib
import numpy as np
from logic.runtime.bullet_creation import BatchedBulletBuilder, EncodingBatchedBulletBuilder
from logic.danmaku.example import ExampleDanmaku


class DanmakuEncoder:
    __instance = None
    __cnt = 0
    def __init__(self):
        if type(BatchedBulletBuilder.instance) != EncodingBatchedBulletBuilder:
            EncodingBatchedBulletBuilder()
        self.builder = BatchedBulletBuilder.instance
        DanmakuEncoder.__instance = self
        self.n_args = len(EncodingBatchedBulletBuilder.encoding_scheme)

    def class_to_ps(self, cls, L=200, augmentation=False, onehot=False):
        if not augmentation:
            danmaku = cls()
        else:
            danmaku = cls(augmentation=True)
        danmaku.timeout = 100000
        while self.builder.api_cnt < L:
            danmaku.update()
            self.builder.update()
        if not onehot:
            param_seq = np.array(self.builder.collect()[:L])
            param_seq[:, 0] = (param_seq[:, 0] + 0.5) / 15
            param_seq[:, 1] = (param_seq[:, 1] + 0.5) / 6
        else:
            oridata = np.array(self.builder.collect()[:L])
            param_seq = np.zeros([L, 34], dtype=float)
            for t in range(L):
                btype = int(oridata[t][0] + 0.5)
                color= int(oridata[t][1] + 0.5)
                param_seq[t, btype] = 1.
                param_seq[t, color + 15] = 1.
            param_seq[:, 21:] = oridata[:, 2:]
            # btypes = oridata
        return param_seq

    @staticmethod
    def inst():
        if DanmakuEncoder.__instance is None:
            DanmakuEncoder()
        return DanmakuEncoder.__instance

def get_classes(path):
    res = []
    for root, _, files in os.walk(path):
        for fname in files:
            if fname[-3:] != '.py':
                continue
            module = importlib.import_module('data.code.' + fname[:-3])
            classes = inspect.getmembers(module, inspect.isclass)
            for _, cls in classes:
                if cls.__base__.__name__ == 'Danmaku':
                    res.append(cls)
    return res

def codes_to_param_seq(path, length=500, onehot=False, out_path='data/mat'):
    cnt = 0
    classes = get_classes(path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for cls in classes:
        print('Encoding ', cls)
        out_fname = out_path + '/D%d' % cnt
        param_seq = DanmakuEncoder.inst().class_to_ps(cls, onehot=onehot, L=length)
        np.save(out_fname, param_seq)
        cnt += 1
    print('Finish')


if __name__ == '__main__':
    param_seq = DanmakuEncoder.inst().class_to_ps(ExampleDanmaku, L=64)
    np.save('./data/example.npy', param_seq)
