from logic.objects.bullet_enums import *
from logic.danmaku.common_shooters import *
from logic.runtime.bullet_creation import BatchedBulletBuilder, BulletBuilder
from data.code.augmentation import *
from random import shuffle, uniform


class SegmentShooter(PeriodicShooter):
    def __init__(self, itv0, itv1, T, radius, ai, ad, ways, **cargs):
        super().__init__(itv1)
        self.itv0 = itv0
        self.itv1 = round(itv1)
        self.T = round(T)
        self.radius = radius
        self.ad = ad
        self.ways = ways
        self.cargs = cargs
        self.angle = ai

    def shot(self):
        if (self.cnt % self.itv0) // self.itv1 < self.T:
            BatchedBulletBuilder.instance.create(
                angle=self.angle, radius=self.radius, ways=self.ways, **self.cargs
            )
        self.angle += self.ad


class D1(Danmaku):
    def __init__(self, augmentation=False):
        super(D1, self).__init__()
        sargs1 = {
            'itv': 90, 'speed': 3.2, 'burst': 2.2, 'decay': 0.6, 'snipe': True, 'ways': 32,
            'btype': BulletTypes.STAR, 'color': BulletColors.RED
        }
        sargs2 = {
            'itv0': 41, 'itv1': 2, 'T': 3, 'radius': 48, 'ai': 0, 'ad': 4.9, 'speed': 1.6, 'ways': 12,
            'btype': BulletTypes.STAR, 'color': BulletColors.BLUE
        }
        sargs3 = {
            'itv0': 23, 'itv1': 3, 'T': 3, 'radius': 40, 'ai': 10, 'ad': -2.8, 'speed': 2.5, 'ways': 10,
            'btype': BulletTypes.STAR, 'color': BulletColors.YELLOW
        }
        if augmentation:
             sargs1, sargs2, sargs3 = transform_dicts(sargs1, sargs2, sargs3)
        self.shooters.add(SpiningShooter(**sargs1))
        self.shooters.add(SegmentShooter(**sargs2), SegmentShooter(**sargs3))


class D2(Danmaku):
    def __init__(self, augmentation=False):
        super(D2, self).__init__()
        self.itv, self.ways = 16, 3
        self.ad1, self.ad2 = 27, -11
        self.colors = [BulletColors.RED, BulletColors.YELLOW, BulletColors.BLUE]
        self.angle1, self.angle2 = 90, 13
        self.r1, self.r2 = 60, 25
        self.cargs = {
            'speed': 1.7, 'burst': 7.3, 'ways': 12, 'decay': 0.8, 'radius': 25,
            'bend': 180, 'btype': BulletTypes.STAR
        }
        if augmentation:
            self.itv, self.ways = transform_multi(self.itv, self.ways)
            self.ad1, self.ad2 = transform_multi(self.ad1, self.ad2)
            shuffle(self.colors)
            self.angle1, self.angle2 = transform_multi(self.angle1, self.angle2)
            self.r1, self.r2 = transform_multi(self.r1, self.r2)
            self.cargs, = transform_dicts(self.cargs)
        self.cid = 0

    def act(self):
        if self.cnt % round(self.itv) == 0:
            factor = 1
            for ab in univals(0, 360, round(self.ways + 1))[:-1]:
                BatchedBulletBuilder.instance.create(
                    rho=self.r1, theta=self.angle1 + ab, angle=factor * self.angle2,
                    color=self.colors[self.cid], **self.cargs
                )
                factor = -factor
            self.angle1 += self.ad1
            self.angle2 += self.ad2
            self.cid += 1
            self.cid %= len(self.colors)


class D3(Danmaku):
    def __init__(self, augmentation=False):
        super(D3, self).__init__()
        self.itv1, self.itv2 = 20, 10
        self.p1_ways = 3
        self.p1_angle, self.p2_angle = 0, -30
        self.ad1, self.ad2 = 37, 45
        self.p2_theta = 120
        self.sargs1 = {
            'strlen': 27, 'span': 60, 'ways': 6, 'speed': 2.46,
            'btype': BulletTypes.DROP, 'color': BulletColors.RED
        }
        self.sargs2 = {
            'strlen': 60, 'span': 72, 'speed': 2.1, 'rho': 75, 'ways': 4,
            'btype': BulletTypes.B_STAR, 'color': BulletColors.GREEN
        }
        if augmentation:
            self.itv1, self.itv2 = transform_multi(self.itv1, self.itv2)
            self.p1_angle = uniform(0, 360)
            self.p2_angle += bounded_gauss(sigma=5)
            self.ad1, self.ad2 = transform_multi(self.ad1, self.ad2)
            self.p2_theta = transform_single(self.p2_theta)
            self.sargs1, self.sargs2 = transform_dicts(self.sargs1, self.sargs2)

    def act(self):
        if self.cnt % round(self.itv1) == 0:
            for ab in univals(0, 360, (self.p1_ways + 1))[:-1]:
                BatchedBulletBuilder.instance.create(
                    angle=ab + self.p1_angle, **self.sargs1
                )
            self.p1_angle += self.ad1
        if self.cnt % round(self.itv2) == 7:
            BatchedBulletBuilder.instance.create(theta=self.p2_theta, angle=-self.p2_angle, **self.sargs2)
            BatchedBulletBuilder.instance.create(theta=-self.p2_theta, angle=self.p2_angle, **self.sargs2)
            self.p2_angle += self.ad2


class D4(Danmaku):
    # itv = 8
    # R = 56
    # ways0, ways1 = 3, 6
    # as0, aa = 0.8, -0.02
    # as1 = 4.5
    # cargs = {
    #     'burst': 3.2, 'decay': 0.75, 'speed': 1.8, 'rho': 46, 'ways': 5,
    #     'btype': BulletTypes.CORN, 'color': BulletColors.YELLOW
    # }
    def __init__(self, augmentation=False):
        super(D4, self).__init__()
        self.itv, self.n = 8, 3
        self.as0, self.aa, self.as1 = 0.8, -0.02, 4.5
        self.sargs = {
            'burst': 3.2, 'decay': 0.75, 'speed': 1.8, 'rho': 56, 'ways': 6,
            'btype': BulletTypes.CORN, 'color': BulletColors.YELLOW
        }
        if augmentation:
            self.as0, self.aa, self.as1 = transform_multi(self.as0, self.aa, self.as1)
            self.sargs, = transform_dicts(self.sargs)

    def act(self):
        if self.cnt % round(self.itv) == 0:
            zdire = self.as0 * self.cnt + 0.5 * self.aa * self.cnt **2
            for ab in univals(0, 360, (self.n+1))[:-1]:
                BatchedBulletBuilder.instance.create(
                    theta=zdire + ab, angle=self.as1 * self.cnt, **self.sargs
                )


class D5(Danmaku):
    def __init__(self, augmentation=False):
        super(D5, self).__init__()
        self.itv1, self.bend_spin = 7, 14
        self.sargs1 = {
            'speed': 2.0, 'burst': 7.6, 'decay': 2.7, 'ways': 12, 'radius': 27, 'btype': BulletTypes.STAR
        }
        self.itv2, self.delay = 85, 28
        self.p2_theta = 125
        self.sargs2 = {
            'itv': 6, 'rho': 115, 'speed': 3.4, 'span': 135, 'ways': 5, 'credit': 7, 'btype': BulletTypes.KUNAI
        }
        self.colors = [
            BulletColors.RED, BulletColors.YELLOW, BulletColors.BLUE, BulletColors.PUEPLE, BulletColors.GREEN
        ]
        if augmentation:
            self.itv1, self.bend_spin = transform_multi(self.itv1, self.bend_spin)
            self.itv2, self.delay, self.p2_theta = transform_multi(self.itv2, self.delay, self.p2_theta)
            self.sargs1, self.sargs2 = transform_dicts(self.sargs1, self.sargs2)
            shuffle(self.colors)
        self.clrid = 0
        self.bend = 0.0

    def act(self):
        if self.cnt % round(self.itv1) == 0:
            BatchedBulletBuilder.instance.create(
                bend=self.bend, color=self.colors[self.clrid], **self.sargs1
            )
            self.clrid += 1
            self.clrid %= len(self.colors)
            self.bend += self.bend_spin
            if self.bend >= 360 or self.bend <= 0:
                self.bend_spin *= -1
        if self.cnt % round(self.itv2) == round(self.delay):
            tar = PlayerPosKeeper()
            self.shooters.add(SpiningShooter(theta=self.p2_theta, snipe=tar, **self.sargs2))
            self.p2_theta = -self.p2_theta

