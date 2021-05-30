import numpy as np
from logic.danmaku.basis import Danmaku
from logic.runtime.bullet_creation import BulletBuilder, BatchedBulletBuilder, EncodingBatchedBulletBuilder
from logic.objects.bullet_enums import BulletTypes, BulletColors
from utils.math import Vec2, linear_mapping, bound


def convert_onehot(data):
    res = []
    for row in data:
        item = [0.0] * 10
        item[:7] = row[:7]
        item[7] = int(np.argmax(row[8:22]))
        item[8] = int(np.argmax(row[22:]))
        item[9] = row[7]
        res.append(item)
    return res


def decode_param_seq(data):
    scheme = EncodingBatchedBulletBuilder.encoding_scheme
    delay_cfg = EncodingBatchedBulletBuilder.delay_cfg
    domain = EncodingBatchedBulletBuilder.vrange
    for i in range(len(data)):
        for j in range(len(scheme)):
            vrange = scheme[j]['range']
            if scheme[j]['key'] == 'snipe':
                # print(data[i][j])
                if data[i][j] < 0.15:
                    data[i][j] = False
                elif data[i][j] < 0.25:
                    data[i][j] = True
                else:
                    snipe_domain = (0.25, domain[1])
                    x = bound(data[i][j], snipe_domain[0], snipe_domain[1])
                    data[i][j] = round(linear_mapping(snipe_domain, vrange, x))
                continue
            dtype = scheme[j]['dtype']
            if not dtype in {int, float}:  # Is Enum
                if dtype == BulletTypes:
                    btypeid = round((data[i][j] - 0.5 / 15) * 15)
                    data[i][j] = dtype(btypeid)
                else:
                    colorid = round((data[i][j] - 0.5 / 6) * 6)
                    data[i][j] = dtype(colorid)
            else:
                x = bound(data[i][j], domain[0], domain[1])
                data[i][j] = linear_mapping(domain, vrange, x)
                if dtype == int:
                    data[i][j] = round(data[i][j])
        if data[i][-1] < delay_cfg['bottom']:
            data[i][-1] = 0
        else:
            delay_domain = (delay_cfg['bottom'], domain[1])
            vrange = delay_cfg['range']
            data[i][-1] = round(linear_mapping(delay_domain, vrange, data[i][-1]))


class CMDanmaku(Danmaku):
    def __init__(self, fname):
        super(CMDanmaku, self).__init__(1800)
        self.data = None
        self.load(fname)

    def finish(self):
        return len(self.data) == 0

    def act(self):
        while self.data[-1][-1] == 0:
            paras = self.data.pop()
            BulletBuilder.instance.create(
                pos=Vec2(paras[0], paras[1]), angle=paras[2], speed=paras[3], frac=paras[4], burst=paras[5],
                snipe=paras[6], btype=BulletTypes(paras[7]), color=BulletColors(paras[8])
            )
            if len(self.data) == 0:
                break
        if len(self.data):
            self.data[-1][-1] = self.data[-1][-1] - 1

    def load(self, fname, onehot=True):
        self.data = np.load(fname)
        if onehot:
            self.data = convert_onehot(self.data)
        if type(self.data) == np.ndarray:
            self.data = self.data.tolist()
        self.data.reverse()
        for i in range(len(self.data)):
            self.data[i][0] = self.data[i][0] * 384.0 - 192.0   # x
            self.data[i][1] = self.data[i][1] * 448.0 - 224.0   # y
            self.data[i][2] = self.data[i][2] * 400.0 - 20.0    # angle
            self.data[i][3] = self.data[i][3] * 6.0 + 0.5       # speed
            self.data[i][4] = self.data[i][4] * 4.0             # frac
            self.data[i][5] = self.data[i][5] * 20.0 - 0.2      # burst
            self.data[i][6] = int(self.data[i][6] * 100 - 2.0)      # snipe
            self.data[i][9] = int(self.data[i][9] * 100)      # delay
            print(self.data[i])
        print('=================================================')


class BatchedCMDanmaku(Danmaku):
    def __init__(self):
        super().__init__(1800)
        self.data = None
        self.scheme = EncodingBatchedBulletBuilder.encoding_scheme

    def act(self):
        if not self.data:
            return
        kwparas = {}
        while self.data[-1][-1] == 0:
            params = self.data.pop()
            for i in range(len(self.scheme)):
                kw = self.scheme[i]['key']
                kwparas[kw] = params[i]
            BatchedBulletBuilder.instance.create(**kwparas)
            if len(self.data) == 0:
                break
        if len(self.data):
            self.data[-1][-1] = self.data[-1][-1] - 1

    def load_file(self, fname):
        self.load(np.load(fname))

    def load(self, param_seq):
        self.data = param_seq
        if type(self.data) == np.ndarray:
            self.data = self.data.tolist()
        self.data.reverse()
        decode_param_seq(self.data)
        _, T, _ = self.get_features()
        self.timeout = T

    def get_features(self):
        L, T = len(self.data), sum(item[-1] for item in self.data)
        EF = 0
        for item in self.data:
            EF += max(0, item[-1]-1)
        return L, T + 180, EF
