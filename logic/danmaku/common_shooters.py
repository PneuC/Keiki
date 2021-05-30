from logic.danmaku.basis import *
from logic.danmaku.basis import PeriodicShooter, PlayerPosKeeper
from logic.danmaku.configurations import *
from logic.runtime.bullet_creation import BulletBuilder, BatchedBulletBuilder


class SpiningShooter(PeriodicShooter):
    def __init__(self, itv=1, delay=0, ad=0.0, ia=0.0, credit=-1, speed_range=None, clr_sheme=None, snipe=False, **cargs):
        super().__init__(itv, delay)
        self.dire = ia
        self.ad = ad
        self.snipe = snipe
        if type(credit) == float:
            credit = round(credit)
        self.credit = credit
        self.cargs = cargs
        if type(speed_range) == tuple and len(speed_range) == 2:
            self.i = 0
            self.n = credit
            self.speeds = univals(speed_range[0], speed_range[1], self.n)
        else:
            self.speeds = None
        if type(clr_sheme) in {tuple, list}:
            self.clrid = 0
            self.clr_sheme = clr_sheme
        else:
            self.clr_sheme = None

    def finish(self):
        return self.credit == 0

    def shot(self):
        if not self.clr_sheme is None:
            self.cargs['color'] = self.clr_sheme[self.clrid]
            self.clrid += 1
            self.clrid %= len(self.clr_sheme)
        if self.speeds is None:
            BatchedBulletBuilder.instance.create(
                angle=self.dire, snipe=self.snipe, **self.cargs
            )
        else:
            BatchedBulletBuilder.instance.create(
                angle=self.dire, snipe=self.snipe, speed=self.speeds[self.i], **self.cargs
            )
        if self.speeds is not None:
            self.i += 1
        self.dire += self.ad
        self.credit -= 1


class FakeLaser(PeriodicShooter):
    def __init__(self, itv, n, **cargs):
        super(FakeLaser, self).__init__(itv)
        self.credit = n
        self.cargs = cargs

    def finish(self):
        return self.credit <= 0

    def shot(self):
        BulletBuilder.instance.create(**self.cargs)
        self.credit -= 1


class TrapezoidShooter(PeriodicShooter):
    def __init__(self, n0=1, t=1, diffusion=18.0, itv=5, delay=0, snipe=False, **kwargs):
        super().__init__(itv, delay)
        if snipe:
            self.snipe = PlayerPosKeeper()
        else:
            self.snipe = False
        self.n = n0
        self.credits = t
        self.diffusion = diffusion
        self.common_kwargs = kwargs

    def finish(self):
        return self.credits <= 0

    def shot(self):
        BatchedBulletBuilder.instance.create(
            span=self.diffusion * (self.n - 1), ways=self.n, snipe=self.snipe, **self.common_kwargs
        )
        self.n += 1
        self.credits -= 1