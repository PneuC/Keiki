from logic.objects.bullet_enums import BulletTypes, BulletColors
from logic.runtime.bullet_creation import BatchedBulletBuilder
from random import uniform, shuffle
from data.code.augmentation import *
from logic.danmaku.basis import Danmaku, PeriodicShooter, PlayerPosKeeper
from logic.danmaku.common_shooters import SpiningShooter, TrapezoidShooter
from logic.danmaku.configurations import univals


class D1(Danmaku):
    # Part1 parameters
    P1_itv = 4
    P1_ss = -13
    P1_ia, P1_cr, P1_bend = -45, 28, 30
    P1_burst0, P1_burst1 = 8.8, 2.5
    P1_bend0, P1_bend1 = 70, 114
    P1_cargs = {'decay': 0.8, 'speed': 1.4, 'btype': BulletTypes.OFUDA}
    # Part2
    P2_itv = 90
    P2_ways = 8
    P2_kwargs = {
        'speed': 2.0, 'burst': 9.0, 'decay': 1.8, 'snipe': True,
        'btype': BulletTypes.M_JADE, 'color': BulletColors.WHITE
    }

    def __init__(self, augmentation=False):
        super().__init__()
        burst1, burst2 = 8.8, 2.5
        bend1, bend2 = 70, 114
        sargs1 = {
            'itv': 4, 'ia': -45, 'ad': -13, 'radius': 28, 'decay': 0.8, 'ways': 2,
            'speed': 1.4, 'btype': BulletTypes.OFUDA
        }
        sargs2 = {
            'itv': 90, 'delay': 38, 'ways': 8, 'speed': 2.0, 'burst': 9.0, 'decay': 1.8, 'snipe': True,
            'btype': BulletTypes.M_JADE, 'color': BulletColors.WHITE
        }
        if augmentation:
            burst1, burst2 = transform_multi(burst1, burst2)
            bend1, bend2 = transform_multi(bend1, bend2)
            sargs1, sargs2 = transform_dicts(sargs1, sargs2)
        self.shooters.add(
            SpiningShooter(
                bend=bend1, burst=burst1, color=BulletColors.BLUE, **sargs1
            ),
            SpiningShooter(
                bend=bend2, burst=burst2, color=BulletColors.RED, **sargs1
            ),
            SpiningShooter(**sargs2)
        )


class SpiningStar(PeriodicShooter):
    def __init__(self, itv, smid, sedge, ways, n, diffusion, ad, credit=-1, **cargs):
        super().__init__(itv)
        self.ways = ways
        self.angle = 0
        self.ad = ad
        self.diffusion = diffusion
        self.credit = round(credit)
        self.n = round(n)
        self.speeds = univals(smid, sedge, self.n)
        self.cargs = cargs
        self.tar = PlayerPosKeeper()

    def shot(self):
        if self.credit == 0:
            self.dead = True
            return
        BatchedBulletBuilder.instance.create(
            angle=self.angle, ways=self.ways, speed=self.speeds[0], snipe=self.tar, **self.cargs
        )
        for i in range(1, self.n):
            BatchedBulletBuilder.instance.create(
                angle=self.angle + i * D2.diffusion, speed=self.speeds[i], snipe=self.tar, ways=self.ways, **self.cargs
            )
            BatchedBulletBuilder.instance.create(
                angle=self.angle - i * D2.diffusion, speed=self.speeds[i], snipe=self.tar, ways=self.ways, **self.cargs
            )
        self.angle += self.ad
        self.credit -= 1


class D2(Danmaku):
    itv = 21
    smin = 1.5
    smax = 2.1
    ways = 5
    diffusion = 6
    n = 3
    credit = 3
    ad = 57
    cargs = {
        'btype': BulletTypes.CORN,
        'color': BulletColors.RED
    }

    def __init__(self, augmentation=False):
        super().__init__()
        self.itv = 21
        self.args = {
            'itv': 21, 'smid': 2.1, 'sedge': 1.5, 'ways': 5, 'diffusion': 6, 'n': 3, 'credit': 3,
            'ad': 47, 'btype': BulletTypes.CORN, 'color': BulletColors.RED
        }
        if augmentation:
            self.itv = transform_single(self.itv)
            self.args, = transform_dicts(self.args)

    def act(self):
        if self.cnt % round(self.itv * self.args['credit']) == 0:
            self.shooters.add(SpiningStar(**self.args))


class D3(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        self.i = 0
        self.itv = 70
        self.colors = [BulletColors.RED, BulletColors.BLUE, BulletColors.GREEN]
        self.args = {
            'itv': 28, 'smid': 1.8, 'sedge': 1.2, 'ways': 4, 'diffusion': 5, 'n': 4,
            'ad': 40, 'credit': 3, 'btype': BulletTypes.KUNAI
        }
        if augmentation:
            self.itv = transform_single(self.itv)
            shuffle(self.colors)
            self.args, = transform_dicts(self.args)

    def act(self):
        if self.cnt % round(self.itv) == 0:
            self.shooters.add(SpiningStar(color=self.colors[self.i], **self.args))
            self.i += 1
            self.i %= len(self.colors)


class D4(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        self.itv0, self.itv1, self.itv2 = 140, 5, 30
        self.n1, self.n2 = 12, 9
        self.ua, self.diffusion = 120, 12
        self.cargs1 = {
            'speed': 1.8, 'burst': 4.2, 'decay': 3.0, 'btype': BulletTypes.BULLET, 'color': BulletColors.WHITE
        }
        cargs2 = {
            'speed': 1.28, 'burst': 8.4, 'decay': 1.75, 'snipe': True, 'span': 60, 'ways': 5,
            'btype': BulletTypes.M_JADE, 'color': BulletColors.WHITE
        }
        delay, td = 80, 30
        if augmentation:
            self.itv0, self.itv1, self.itv2 = transform_multi(self.itv0, self.itv1, self.itv2)
            self.n1, self.n2 = transform_multi(self.n1, self.n2)
            self.ua, self.diffusion = transform_multi(self.ua, self.diffusion)
            self.cargs1, cargs2 = transform_dicts(self.cargs1, cargs2)
            delay, td = transform_multi(delay, td)
        self.shooters.add(
            SpiningShooter(delay=delay, itv=self.itv0, **cargs2),
            SpiningShooter(delay=delay + td, itv=self.itv0, **cargs2)
        )

    def act(self):
        itv0, itv1 = round(self.itv0), round(self.itv1)
        n1 , n2 = round(self.n1), round(self.n2)
        t = self.cnt % itv0
        w = t // itv1 + 1
        angles0 = (0, self.ua, -self.ua)
        if t % itv1 == 0 and w < n1:
            for a0 in angles0:
                BatchedBulletBuilder.instance.create(
                    angle=a0, ways=w, span=(w - 1) * self.diffusion, **self.cargs1
                )
        w -= n1
        if t % itv1 == 0 and 0 < w < n2:
            for a0 in angles0:
                BatchedBulletBuilder.instance.create(
                    angle=a0, ways=w, span=(w - 1) * self.diffusion, **self.cargs1
                )
