from logic.objects.bullet_enums import BulletTypes, BulletColors
from logic.runtime.bullet_creation import BatchedBulletBuilder
from random import randrange, uniform, shuffle
from data.code.augmentation import *
from logic.danmaku.basis import Danmaku
from logic.danmaku.common_shooters import SpiningShooter
from logic.danmaku.configurations import univals



class D1(Danmaku):
    def __init__(self, augmentation=False):
        super(D1, self).__init__(move_config=(-1, 0))
        ab, ad = 0, -9
        r1 = 90
        sargs1 = {
            'itv': 15, 'span': 33, 'ways':2, 'speed': 2.1,
            'btype': BulletTypes.S_JADE, 'color': BulletColors.BLUE
        }
        smin, smax = 2.2, 3.0
        sargs2 = {
            'itv': 120, 'delay': 80, 'snipe': True, 'ways': 3,
            'btype': BulletTypes.G_JADE, 'color': BulletColors.BLUE
        }
        if augmentation:
            ab = uniform(0, 360)
            ad, r1 = transform_multi(ad, r1)
            smin, smax = transform_multi(smin, smax)
            sargs1, sargs2 = transform_dicts(sargs1, sargs2)
        r2 = r1/ 3 ** 0.5
        shooters0 = [
            SpiningShooter(
                ia=agl + ab, ad=ad, bend=180, radius=r1, **sargs1
            ) for agl in (0, 120, 240)
        ]
        shooters1 = [
            SpiningShooter(
                ia=agl + ab, ad=-ad, bend=180, radius=r2, **sargs1
            ) for agl in (60, 180, 300)
        ]
        shooters2 = [SpiningShooter(speed=spd, **sargs2) for spd in univals(smin, smax, 3)]
        self.shooters.add(*shooters0, *shooters1, *shooters2)


class D2(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        rho1, rho2 = 80, 40
        theta, ad = 105, 5.7
        sargs1 = {
            'itv': 15, 'speed': 1.1, 'span': 36, 'ways': 2, 'radius': 36,
            'btype': BulletTypes.CORN, 'color': BulletColors.WHITE
        }
        sargs2 = {
            'itv': 120, 'delay': 75, 'speed': 0.96, 'snipe': True, 'rho': 40,
            'btype':BulletTypes.M_JADE, 'color': BulletColors.WHITE
        }
        if augmentation:
            rho1, rho2 = transform_multi(rho1, rho2)
            theta, ad = transform_multi(theta, ad)
            sargs1, sargs2 = transform_dicts(sargs1, sargs2)
        shooters0 = [
            SpiningShooter(
                rho=rho1, ia=agl, ad=ad, theta=theta, bend=180, **sargs1
            ) for agl in (0, 120, 240)
        ]
        shooters1 = [
            SpiningShooter(
                rho=rho1, ia=agl, ad=-ad, theta=-theta, bend=180, **sargs1
            ) for agl in (0, 120, 240)
        ]
        shooters2 = [
            SpiningShooter(
                rho=rho2, ia=agl, ad=ad, bend=180, **sargs1
            ) for agl in (0, 180)
        ]
        shooters3 = [
            SpiningShooter(
                rho=rho2, ia=agl, ad=-ad, bend=180, **sargs1
            ) for agl in (90, 270)
        ]
        self.shooters.add(*shooters0, *shooters1, *shooters2, *shooters3)
        self.shooters.add(SpiningShooter(**sargs2))


class D3(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        self.rho1, self.rho2 = 80, 105
        self.ways1, self.ways2 = 3, 2
        self.theta = 105
        self.sargs = {
            'itv': 8, 'radius': 36, 'credit': 7, 'speed': 1.76,
            'btype': BulletTypes.BULLET, 'color': BulletColors.BLUE
        }
        self.p1_2_itv0 = 27
        self.p1_2_angle = 0
        self.ad = -16.5
        s31, s32, ways3 = 2.25, 1.98, 21
        sargs2 = {
            'itv': 50, 'delay': 30, 'burst': 3.4, 'decay': 1.6, 'snipe': True,
            'btype':BulletTypes.SQUAMA, 'color': BulletColors.PUEPLE
        }
        if augmentation:
            self.rho1, self.rho2 = transform_multi(self.rho1, self.rho2)
            self.ways1, self.ways2 = transform_multi(self.ways1, self.ways2)
            self.theta, self.p1_2_itv0, self.p1_2_angle, self.ad = \
                transform_multi(self.theta, self.p1_2_itv0, self.p1_2_angle, self.ad)
            s31, s32, ways3 = transform_multi(s31, s32, ways3)
            self.sargs, sargs2 = transform_dicts(self.sargs, sargs2)
        self.shooters.add(
            SpiningShooter(speed=s31, ways=ways3, **sargs2),
            SpiningShooter(ia=180 / round(ways3+1), speed=s32, ways=ways3+1, **sargs2)
        )

    def act(self):
        if self.cnt % round(self.p1_2_itv0) == 0:
            self.shooters.add(
                SpiningShooter(
                    ia=self.p1_2_angle, ways=self.ways1, bend=180, rho=self.rho1, theta=self.theta, **self.sargs
                ),
                SpiningShooter(
                    ia=-self.p1_2_angle, ways=self.ways1, bend=180, rho=self.rho1, theta=-self.theta, **self.sargs
                ),
                SpiningShooter(
                    ia=self.p1_2_angle, ways=self.ways2, bend=180, rho=self.rho2, **self.sargs
                ),
                SpiningShooter(
                    ia=-self.p1_2_angle, ways=self.ways2, bend=180, rho=self.rho2, **self.sargs
                )
            )
            self.p1_2_angle += self.ad


class D4(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        cargs = {'itv': 12, 'ways': 5, 'ad': -19}
        sargs1 = {
            'speed': 2.2, 'bend': 60, 'radius': 30,
            'btype': BulletTypes.CORN, 'color': BulletColors.WHITE
        }
        sargs2 = {'bend': -75, 'radius': 50, 'btype': BulletTypes.R_JADE, 'color': BulletColors.BLUE}
        sargs3 = {'bend': 75, 'radius': 70, 'btype': BulletTypes.R_JADE, 'color': BulletColors.RED}
        s1, s2 = 1.5, 2.2
        if augmentation:
            cargs, sargs1, sargs2, sargs3 = transform_dicts(cargs, sargs1, sargs2, sargs3)
            s1, s2 = transform_multi(s1, s2)
        self.shooters.add(
            SpiningShooter(**cargs, **sargs1),
            SpiningShooter(speed=s1, **cargs, **sargs2),
            SpiningShooter(speed=s2, **cargs, **sargs2),
            SpiningShooter(speed=s1, **cargs, **sargs3),
            SpiningShooter(speed=s2, **cargs, **sargs3),
        )


class D5(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        decay0, decay1 = 0.24, 0.3
        ab, ad = 0, 7.4
        cargs = {
            'rho': 75, 'itv': 5, 'ways': 2, 'speed': 0.9, 'burst': 6.6,
            'btype': BulletTypes.BULLET, 'color': BulletColors.RED
        }
        if augmentation:
            decay0, decay1 = transform_multi(decay0, decay1)
            ab, ad = transform_multi(ab, ad)
            cargs, = transform_dicts(cargs)
        self.shooters.add(
            SpiningShooter(ad=ad, theta=135 + ab, decay=decay0, **cargs),
            SpiningShooter(ad=-ad, theta=225+ ab, decay=decay0, **cargs),
            SpiningShooter(ad=-ad, theta=45 + ab, decay=decay1, **cargs),
            SpiningShooter(ad=ad, theta=315 + ab, decay=decay1, **cargs)
        )
        self.angle = 0


class D6(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        self.itv = 9
        self.sargs = {
            'speed': 0.84, 'burst': 7.0, 'decay': 0.17, 'strlen': 84, 'ways':6, 'snipe': True,
            'btype': BulletTypes.BULLET, 'color': BulletColors.RED
        }
        self.angle, self.ad, self.span = 0, 10.9, 32
        strlen, n = 84, 6
        sargs2 = {
            'itv': 55, 'span': 240, 'ways': 11, 'speed': 1.13, 'snipe': True,
            'btype': BulletTypes.BULLET, 'color': BulletColors.RED
        }
        if augmentation:
            self.sargs, sargs2 = transform_dicts(self.sargs, sargs2)
            self.itv = transform_single(self.itv)
            self.angle, self.ad, self.span = transform_multi(self.angle, self.ad, self.span)
            strlen, n = transform_multi(strlen, n)
        for rho in univals(-strlen / 2, strlen / 2, round(n)):
            theta = 90 if rho > 0 else 270
            self.shooters.add(
                SpiningShooter(rho=abs(rho), theta=theta, ia=180-rho, **sargs2)
            )

    def act(self):
        if self.cnt % round(self.itv) == 0:
            BatchedBulletBuilder.instance.create(angle=self.angle, **self.sargs)
            self.angle += self.ad
            if abs(self.angle) > self.span:
                self.ad = -self.ad

