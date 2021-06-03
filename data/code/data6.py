from random import uniform
from logic.danmaku.basis import Danmaku
from logic.danmaku.common_shooters import TrapezoidShooter, SpiningShooter
from logic.danmaku.configurations import *
from data.code.augmentation import *
from logic.objects.bullet_enums import BulletTypes, BulletColors
from logic.runtime.bullet_creation import BatchedBulletBuilder


class D1(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        self.itv, self.d1, self.d2 = 180, 40, 100
        self.sargs1 = {
            'speed': 1.9, 'burst': 8.0, 'decay': 2.5, 'itv': 4, 'diffusion': 1.45, 't': 8,
            'btype': BulletTypes.BULLET, 'color': BulletColors.WHITE
        }
        self.sargs2 = {
            'speed': 1.5, 'burst': 8.5, 'decay': 2, 'n0': 9, 't': 7, 'itv': 7,
            'diffusion': 18, 'btype': BulletTypes.M_JADE
        }
        self.args = {'snipe': True, 'btype': BulletTypes.ELLIPSE, 'color': BulletColors.PUEPLE}
        self.s1, self.s2 = 0.9, 1.05
        if augmentation:
            self.itv, self.d1, self.d2 = transform_multi(self.itv, self.d1, self.d2)
            self.sargs1, self.sargs2, self.args = transform_dicts(self.sargs1, self.sargs2, self.args)
            self.s1, self.s2 = transform_multi(self.s1, self.s2)

    def act(self):
        t = self.cnt % round(self.itv)
        if t == 0:
            self.shooters.add(TrapezoidShooter(**self.sargs1))
        if t == round(self.d1):
            self.shooters.add(TrapezoidShooter(**self.sargs2))
        if t == round(self.d2):
            speeds = univals(self.s1, self.s2, 3)
            for s in speeds:
                BatchedBulletBuilder.instance.create(speed=s, **self.args)


class D2(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        self.itv1, self.td1, self.td2 = 120, 176, 40
        self.sargs1 = {
            'speed': 2.8, 'burst': 13.5, 'decay': 3.55, 't': 8, 'itv': 3, 'snipe': True,
            'diffusion': 1.68, 'btype': BulletTypes.BULLET
        }
        self.sargs2 = {
            'speed': 1.5, 'burst': 8.5, 'decay': 2, 'n0': 9, 't': 2, 'itv': 10,
            'diffusion': 21, 'btype': BulletTypes.M_JADE
        }
        if augmentation:
            self.itv1, self.td1, self.td2 = transform_multi(self.itv1, self.td1, self.td2)
            self.sargs1, self.sargs2 = transform_dicts(self.sargs1, self.sargs2)

    def act(self):
        itv1, td1, td2 = round(self.itv1), round(self.td1), round(self.td2)
        if self.cnt % itv1 == 0 or self.cnt % (2 * itv1) == td1:
            self.shooters.add(TrapezoidShooter(**self.sargs1))
        if self.cnt % itv1 == td2 or self.cnt % (2 * itv1) == td1 + td2:
            self.shooters.add(TrapezoidShooter(**self.sargs2))


class D3(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        self.itv, self.d1, self.d2 = 180, 40, 100
        self.sargs1 = {
            'speed': 1.9, 'burst': 12.8, 'decay': 2.5, 'interval': 4, 'diffusion': 1.15, 't': 8,
            'snipe': True, 'btype': BulletTypes.BULLET
        }
        self.sargs2 = {
            'speed': 2.4, 'burst': 12.8, 'decay': 2.5, 'n0': 9, 't': 8, 'interval': 7,
            'diffusion': 17, 'btype': BulletTypes.M_JADE
        }
        self.args = {'snipe': True, 'btype': BulletTypes.ELLIPSE, 'color': BulletColors.PUEPLE}
        self.s1, self.s2 = 1.15, 1.4
        if augmentation:
            self.itv, self.d1, self.d2 = transform_multi(self.itv, self.d1, self.d2)
            self.sargs1, self.sargs2, self.args = transform_dicts(self.sargs1, self.sargs2, self.args)
            self.s1, self.s2 = transform_multi(self.s1, self.s2)

    def act(self):
        t = self.cnt % round(self.itv)
        if t == 0:
            self.shooters.add(TrapezoidShooter(**self.sargs1))
        if t == round(self.d1):
            self.shooters.add(TrapezoidShooter(**self.sargs2))
        if t == round(self.d2):
            speeds = univals(self.s1, self.s2, 3)
            for s in speeds:
                BatchedBulletBuilder.instance.create(speed=s, **self.args)


class D4(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        ia = 0
        smin, smax, n1 = 1.0, 1.3, 5
        sargs1 = {
            'itv': 67, 'ad': 17, 'ways': 16, 'radius': 15,
            'btype': BulletTypes.BULLET, 'color': BulletColors.WHITE
        }
        self.itv21, self.itv22 = 75, 15
        self.sargs2 = {
            'delay': 22, 'n0': 1, 't': 7, 'interval': 3, 'speed': 2.68, 'burst': 7, 'decay': 1.8,
            'snipe': True,  'diffusion': 2.25, 'btype': BulletTypes.BULLET
        }
        self.p2_ad = 36
        if augmentation:
            ia = uniform(0, 360)
            smin, smax = transform_multi(smin, smax)
            sargs1, self.sargs2 = transform_dicts(sargs1, self.sargs2)
            self.itv21, self.itv22 = transform_multi(self.itv21, self.itv22)
            self.itv21, self.itv22 = round(self.itv21), round(self.itv22)
            self.p2_ad = transform_single(self.p2_ad)
        self.shooters.add(*(SpiningShooter(ia=ia, speed=s, **sargs1) for s in univals(smin, smax, n1)))

    def act(self):
        if (self.cnt % self.itv21) == 0:
            self.shooters.add(TrapezoidShooter(**self.sargs2))
        if (self.cnt % self.itv21) == self.itv22:
            self.shooters.add(
                TrapezoidShooter(**self.sargs2, angle=self.p2_ad * 2),
                TrapezoidShooter(**self.sargs2, angle=-self.p2_ad * 2)
            )
        if (self.cnt % self.itv21) == self.itv22 * 2:
            self.shooters.add(
                TrapezoidShooter(**self.sargs2, angle=self.p2_ad),
                TrapezoidShooter(**self.sargs2, angle=-self.p2_ad)
            )


class D5(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        itv, ways, ad = 130, 49, 1.3
        td, n_laser, laser_itv = 35, 8, 6
        sargs1 = {
            'burst': 8.9, 'decay': 0.46, 'speed': 0.91,
            'btype': BulletTypes.S_JADE, 'color': BulletColors.BLUE
        }
        sargs2 = {
            'burst': 7.5, 'decay': 0.6, 'speed': 2.1, 'radius': 50,
            'btype': BulletTypes.CORN, 'color': BulletColors.BLUE
        }
        if augmentation:
            itv, ways, ad = transform_multi(itv, ways, ad)
            td = round(transform_single(td))
            sargs1, sargs2 = transform_dicts(sargs1, sargs2)
        self.shooters.add(SpiningShooter(itv, ia=360 / round(ways) / 2, ways=ways, ad=ad, **sargs1))
        self.shooters.add(*(
            SpiningShooter(itv, delay=td + laser_itv * i, ways=ways, ad=ad, **sargs2)
            for i in range(n_laser)
        ))


class D6(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        self.itv = 23
        self.r_min, self.r_max = 22, 85
        self.angle, self.ad = 0, 4.3
        rd = 13.5
        self.bend = 80
        self.cargs = {
            'ways': 22, 'speed': 1.8, 'btype': BulletTypes.OFUDA
        }
        if augmentation:
            self.itv = round(transform_single(self.itv))
            self.r_min, self.r_max = transform_multi(self.r_min, self.r_max)
            self.angle, self.ad = transform_multi(self.angle, self.ad)
            rd, self.bend = transform_multi(rd, self.bend)
            self.cargs, = transform_dicts(self.cargs)
        # print(self.cargs['ways'])
        self.r = self.r_min
        self.rd = rd

    def act(self):
        if self.cnt % self.itv == 0:
            BatchedBulletBuilder.instance.create(
                radius=self.r, bend=self.bend, angle=self.angle, color=BulletColors.BLUE, **self.cargs
            )
            BatchedBulletBuilder.instance.create(
                radius=self.r, bend=-self.bend, angle=-self.angle, color=BulletColors.RED, **self.cargs
            )
            if self.r >= self.r_max or self.r < self.r_min:
                self.rd = -self.rd
            self.r += self.rd
            self.angle += self.ad
