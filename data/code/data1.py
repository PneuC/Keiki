from logic.objects.bullet_enums import BulletTypes, BulletColors
from logic.runtime.bullet_creation import BatchedBulletBuilder
from random import randrange, uniform, shuffle
from data.code.augmentation import *
from logic.danmaku.basis import Danmaku
from logic.danmaku.common_shooters import SpiningShooter


class D1(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        itv = 12
        a1, a2, a3 = 180., 15., 100.
        sargs1 = {
            'span': 160., 'ways': 4, 'ad': -28., 'speed': 2.8, 'burst': 3.7, 'decay': 2.0,
            'btype': BulletTypes.G_JADE, 'color': BulletColors.RED
        }
        sargs2 = {
            'span': 45., 'ways': 2, 'ad': -14., 'speed': 2.5,
            'btype': BulletTypes.M_JADE, 'color': BulletColors.BLUE
        }
        sargs3 = {
            'span': 15., 'ways': 3, 'ad': -35., 'speed': 1.8, 'burst': 3.2, 'decay': 1.4,
            'btype': BulletTypes.S_JADE, 'color': BulletColors.PUEPLE
        }
        if augmentation:
            itv += randrange(-2, 4)
            a1, a2, a3 = transform_gl([a1, a2, a3], g_bias=uniform(0, 360))
            sargs1, sargs2, sargs3 = transform_dicts(sargs1, sargs2, sargs3)
        self.shooters.add(
            SpiningShooter(itv, ia=a1, **sargs1),
            SpiningShooter(itv, ia=a2, **sargs2),
            SpiningShooter(itv, ia=a2 + 180, **sargs2),
            SpiningShooter(itv, ia=a3, **sargs3)
        )


class D2(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        ad = 5.85
        sargs1 = {'itv': 6, 'ways': 6, 'ia':35.4, 'speed': 1.77, 'btype': BulletTypes.CORN, 'color': BulletColors.GREEN}
        sargs2 = {
            'itv': 70, 'delay': 53, 'ways': 38, 'speed': 1.43, 'snipe': True,
            'btype': BulletTypes.CORN, 'color': BulletColors.RED
        }
        if augmentation:
            ad += transform_single(ad)
            sargs1, sargs2 = transform_dicts(sargs1, sargs2)
        self.shooters.add(
            SpiningShooter(ad=ad, **sargs1),
            SpiningShooter(ad=-ad, **sargs1),
            SpiningShooter(**sargs2)
        )


class D3(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        colors0 = [
            BulletColors.RED, BulletColors.YELLOW, BulletColors.GREEN,
            BulletColors.WHITE, BulletColors.BLUE, BulletColors.PUEPLE
        ]
        colors1 = [BulletColors.WHITE, BulletColors.GREEN, BulletColors.BLUE]
        ways1, ways2, ways3= 5, 6, 3
        span1, span2, span3 = 45, 65, 42
        ia = 35
        sargs = {'itv':8, 'ad': 9.3, 'speed': 1.98, 'btype': BulletTypes.CORN}
        if augmentation:
            ia += uniform(-15, 15)
            ways1, ways2, ways3 = transform_multi(ways1, ways2, ways3)
            span1, span2, span3 = transform_multi(span1, span2, span3)
            shuffle(colors0), shuffle(colors1)
        self.shooters.add(
            SpiningShooter(ia=ia, ways=ways1, span=span1, clr_sheme=colors0, **sargs),
            SpiningShooter(ia=ia + 180, ways=ways2, span=span2, clr_sheme=colors1, **sargs),
            SpiningShooter(ia=ia + 90, ways=ways2, span=span2, clr_sheme=colors1, **sargs),
            SpiningShooter(ia=ia + 270, ways=ways3, span=span3, clr_sheme=colors1, **sargs)
        )


class D4(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        r11, r12 = 18, 35
        self.r21, self.r22 = 24, 45
        sargs1 = {
            'itv':37, 'delay': 9, 'speed': 2.15, 'snipe': True, 'ways': 22,
            'btype': BulletTypes.S_JADE, 'color': BulletColors.GREEN
        }
        self.itv2, self.start2, self.itv3 = 37, 23, 75
        self.p2_ab, self.p3_ab, self.p3_strlen = -4.5, 9.0, 46
        self.sargs2 = {
            'speed': 2.8, 'snipe': True, 'ways': 18,
            'btype': BulletTypes.S_JADE, 'color': BulletColors.WHITE
        }
        self.sargs3 = {
            'speed': 1.68, 'ways': 12, 'span': 175,
            'btype': BulletTypes.M_JADE, 'color': BulletColors.BLUE
        }
        if augmentation:
            sargs1, self.sargs2, self.sargs3 = transform_dicts(sargs1, self.sargs2, self.sargs3)
            r11, r12 = transform_multi(r11, r12)
            self.itv2, self.start2, self.itv3 = transform_multi(self.itv2, self.start2, self.itv3)
            self.p2_ab, self.p3_ab, self.p3_strlen = transform_multi(self.p2_ab, self.p3_ab, self.p3_strlen)
            self.r21, self.r22 = transform_multi(self.r21, self.r22)
        self.shooters.add(SpiningShooter(radius=r11, **sargs1), SpiningShooter(radius=r12, **sargs1))

    def act(self):
        if self.cnt % round(self.itv2) == round(self.start2):
            BatchedBulletBuilder.instance.create(radius=self.r21, **self.sargs2)
            BatchedBulletBuilder.instance.create(radius=self.r22, angle=self.p2_ab, **self.sargs2)
            self.p2_ab = -self.p2_ab
        if self.cnt % round(self.itv3) == 0:
            BatchedBulletBuilder.instance.create(angle=self.p3_ab,  **self.sargs3)
            BatchedBulletBuilder.instance.create(angle=self.p3_ab, strlen=self.p3_strlen,  **self.sargs3)
            self.p3_ab = -self.p3_ab


class D5(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        self.itv1, self.itv2 = 65, 130
        self.sargs1 = {
            'itv': 5, 'ways': 8, 'ad': 5.5, 'credit': 12,
            'btype': BulletTypes.CORN,'color': BulletColors.RED}
        self.sargs2 = {
            'itv': 4, 'delay':40,  'ways': 14, 'ad': -2.1, 'credit': 7,
            'btype': BulletTypes.CORN,'color': BulletColors.WHITE
        }
        self.smin1, self.smax1 = 1.0, 2.8
        self.smin2, self.smax2 = 1.2, 1.9
        # self.n1, self.n2 = 12, 7
        self.p1_angle, self.p2_angle = -22, 8
        self.p1_ad, self.p2_ad = 17, 11
        if augmentation:
            self.itv1, self.itv2 = transform_multi(self.itv1, self.itv2)
            self.sargs1, self.sargs2 = transform_dicts(self.sargs1, self.sargs2)
            mufctr1, mufctr2 = bounded_gauss(sigma=0.05), bounded_gauss(sigma=0.05)
            if mufctr1 > 0:
                self.smin1 *= (1 + mufctr1)
                self.smax1 *= (1 + mufctr1)
            else:
                self.smin1 /= (1 - mufctr1)
                self.smax1 /= (1 - mufctr1)
            if mufctr2 > 0:
                self.smin2 *= (1 + mufctr2)
                self.smax2 *= (1 + mufctr2)
            else:
                self.smin2 /= (1 - mufctr2)
                self.smax2 /= (1 - mufctr2)
            self.p1_angle, self.p2_angle = transform_multi(self.p1_angle, self.p2_angle)
            self.p1_ad, self.p2_ad = transform_multi(self.p1_ad, self.p2_ad)

    def act(self):
        if self.cnt % round(self.itv1) == 0:
            self.shooters.add(
                SpiningShooter(ia=self.p1_angle, speed_range=(self.smin1, self.smax1), **self.sargs1)
            )
            self.p1_angle += self.p1_ad
        if self.cnt % round(self.itv2) == 0:
            self.shooters.add(
                SpiningShooter(ia=self.p2_angle, speed_range=(self.smin2, self.smax2), **self.sargs2)
            )
            self.p2_angle += self.p2_ad


class D6(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        self.itv1, self.itv2 = 275, 225
        self.sargs1 = {
            'itv': 19, 'span': 75, 'speed': 1.9, 'ways': 8, 'credit': 10,
            'btype': BulletTypes.DAGGER, 'color': BulletColors.BLUE
        }
        self.sargs2 = {
            'itv': 12, 'delay': 187, 'span': 9.25, 'ways': 4, 'credit': 12,
            'btype': BulletTypes.KUNAI, 'color': BulletColors.RED
        }
        self.s1, self.s2 = 1.4, 1.8
        span1, span2 = 180, 168
        if augmentation:
            self.itv1, self.itv2 = transform_multi(self.itv1, self.itv2)
            self.sargs1, self.sargs2 = transform_dicts(self.sargs1, self.sargs2)
            self.s1, self.s2 = transform_multi(self.s1, self.s2)
            span1, span2 = transform_multi(span1, span2)
        self.p1_angle = -span1 / 2
        self.p1_ad = span1 / (self.sargs1['credit'] - 1)
        self.p2_angle = span2 / 2
        self.p2_ad = -span2 / (self.sargs2['credit'] - 1)

    def act(self):
        if self.cnt % round(self.itv1) == 0:
            self.shooters.add(SpiningShooter(ad=self.p1_ad, ia = self.p1_angle, **self.sargs1))
            self.p1_angle = -self.p1_angle
            self.p1_ad = - self.p1_ad
        if self.cnt % round(self.itv2) == 0:
            self.shooters.add(
                SpiningShooter(
                    ad=self.p2_ad, ia=self.p2_angle, speed=self.s1, **self.sargs2
                ),
                SpiningShooter(
                    ad=self.p2_ad, ia=self.p2_angle, speed=self.s2, **self.sargs2
                )
            )


class D7(Danmaku):
    def __init__(self, augmentation=False):
        super(D7, self).__init__()
        s0, s1 = 1.58, 2.04
        ia, ad, bend = 4.8, 8.8, 75
        clr0, clr1 = BulletColors.WHITE, BulletColors.YELLOW
        cargs = {
            'itv': 42, 'ways': 16, 'radius': 48, 'burst': 2.6,
            'decay': 1.3, 'btype': BulletTypes.S_JADE
        }
        if augmentation:
            s0, s1 = transform_multi(s0, s1)
            ia, ad, bend = transform_multi(ia, ad, bend)
            cargs, = transform_dicts(cargs)
        halfitv = cargs['itv'] / 2
        self.shooters.add(
            SpiningShooter(ia=ia, ad=ad, speed=s0,bend=bend, color=clr0, **cargs),
            SpiningShooter(ia=ia, ad=ad, speed=s1, bend=bend, color=clr0, **cargs),
            SpiningShooter(delay=halfitv, ia=-ia, ad=-ad, speed=s0, bend=-bend, color=clr1, **cargs),
            SpiningShooter(delay=halfitv, ia=-ia, ad=-ad, speed=s1,bend=-bend, color=clr1, **cargs)
        )


