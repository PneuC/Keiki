from logic.objects.bullet_enums import BulletTypes, BulletColors
from logic.runtime.bullet_creation import BatchedBulletBuilder
from random import uniform, shuffle
from data.code.augmentation import *
from logic.danmaku.basis import Danmaku, PeriodicShooter, PlayerPosKeeper
from logic.danmaku.common_shooters import SpiningShooter, TrapezoidShooter
from logic.danmaku.configurations import univals, sectoral_angles


class D1(Danmaku):
    def __init__(self, augmentation=False):
        self.itv = 5
        self.sa, self.as0, self.ia = -0.015, 0.9, -9
        self.cargs ={
            'speed': 1.4, 'ways': 6, 'btype': BulletTypes.CORN, 'color': BulletColors.WHITE
        }
        if augmentation:
            self.sa, self.as0, self.ia = transform_multi(self.sa, self.as0, self.ia)
            self.cargs, = transform_dicts(self.cargs)
        super().__init__()

    def act(self):
        itv = round(self.itv)
        angle = self.as0 * self.cnt + 0.5 * self.sa * self.cnt * self.cnt + self.ia
        if self.cnt % itv == 0:
            BatchedBulletBuilder.instance.create(angle=angle, **self.cargs)


class D2(Danmaku):
    itv0 = 25
    itv1 = 2
    n = 9
    smin = 3.1
    smax = 4.6
    ways = 9
    ad0 = 21.7
    ad1 = -1.63
    ia = 10
    cargs ={
        'burst': 5.0, 'decay': 1.2, 'btype': BulletTypes.CORN, 'color': BulletColors.RED
    }

    def __init__(self, augmentation=False):
        super().__init__()
        self.i = 0
        self.angle, self.ad = 10, 21.7
        self.smin, self.smax = 3.1, 4.6
        self.cargs = {
            'itv': 2, 'burst': 5.0, 'decay': 1.2, 'ways': 9, 'credit': 8, 'ad': -1.63,
            'btype': BulletTypes.CORN, 'color': BulletColors.RED
        }
        if augmentation:
            mutation = bounded_gauss(sigma=0.025 * (self.smin + self.smax))
            self.smin += mutation
            self.smax += mutation
            self.cargs, = transform_dicts(self.cargs)
            self.ad = transform_single(self.ad)
            self.angle = uniform(0, 360)

    def act(self):
        if self.cnt % D2.itv0 == 0:
            self.shooters.add(
                SpiningShooter(
                    ia=self.angle, speed_range=(self.smin, self.smax), **self.cargs
                )
            )
            self.angle += self.ad


class D3(Danmaku):
    def __init__(self, augmentation=False):
        super().__init__()
        s1, s2 = 1.4, 1.7
        cargs = {
            'itv': 7, 'ia': 13.5, 'ways': 3, 'ad': 34,
            'btype': BulletTypes.SQUAMA
        }
        if augmentation:
            s1, s2 = transform_multi(s1, s2)
            cargs, = transform_dicts(cargs)
        self.shooters.add(
            SpiningShooter(speed=s1, color=BulletColors.BLUE, **cargs),
            SpiningShooter(speed=s2, color=BulletColors.PUEPLE, **cargs)
        )


class D4(Danmaku):
    def __init__(self, augmentation=False):
        super(D4, self).__init__()
        self.itv, self.itv_min = 42, 35
        self.angle, self.span, self.n = 8, 20, 10
        self.ways = 12
        self.color = BulletColors.BLUE
        if augmentation:
            self.itv, self.itv_min = transform_multi(self.itv, self.itv_min)
            self.angle, self.span, self.n = transform_multi(self.angle, self.span, self.n)
            self.itv, self.itv_min, self.n = round(self.itv), round(self.itv_min), round(self.n)
            self.ways = transform_single(self.ways)
        self.speeds = [1.7 + (0.07 * i) ** 2 for i in range(self.n)]


    def act(self):
        if self.cnt % round(self.itv) == 0:
            angles = sectoral_angles(dire=self.angle, span=self.span, ways=self.n)
            for i in range(self.n):
                BatchedBulletBuilder.instance.create(
                     angle=angles[i], speed=self.speeds[i], btype=BulletTypes.S_JADE, ways=self.ways, color=self.color
                )
            self.color = BulletColors.RED if self.color is BulletColors.BLUE else BulletColors.BLUE
            self.speeds.reverse()
            self.angle = -self.angle
            if self.itv > self.itv_min:
                self.itv -= 1

class D5(Danmaku):
    def __init__(self, augmentation=False):
        super(D5, self).__init__()
        self.itv0, self.itv1 = 215, 6
        self.ad, self.n = 1.95, 15
        self.ab, self.abd = 0.0, 3.3
        self.cargs = {
            'speed': 2.4, 'ways': 33, 'btype': BulletTypes.CORN, 'color': BulletColors.BLUE
        }
        if augmentation:
            self.itv0, self.itv1, self.n = transform_multi(self.itv0, self.itv1, self.n)
            self.itv0, self.itv1, self.n = round(self.itv0), round(self.itv1), round(self.n)
            self.ad, self.ab, self.abd = transform_multi(self.ad, self.ab, self.abd)
            self.cargs, = transform_dicts(self.cargs)
        self.angle = self.ad * (self.n - 1)
        self.T = self.n * self.itv1

    def act(self):
        if self.cnt % self.itv0 == 0 or self.cnt % self.itv0 == self.T:
            self.shooters.add(
                SpiningShooter(itv=self.itv1, ia=self.angle + self.ab, credit=self.n, ad=self.ad, **self.cargs)
            )
            self.angle = -self.angle
            self.ad = -self.ad
        if self.cnt % self.itv0 == 0:
            self.ab += self.abd


class D6(Danmaku):
    class MovingCircle(PeriodicShooter):
        def __init__(self, rho, theta, itv, stride, ways, credit, **cargs):
            super().__init__(itv)
            self.rho = rho
            self.theta = theta
            self.stride = stride
            self.ways = ways
            self.credits = round(credit)
            self.cargs = cargs

        def finish(self):
            return self.credits == 0

        def shot(self):
            BatchedBulletBuilder.instance.create(
                rho=self.rho, theta=self.theta, angle=self.theta, ways=self.ways, **self.cargs
            )
            self.rho += self.stride
            self.credits -= 1

    def __init__(self, augmentation=False):
        super(D6, self).__init__()
        self.itv = 160
        self.angle, self.ad = 15, 23
        self.nshtr = 7
        self.cargs = {
            'rho': 0., 'itv': 17, 'speed': 2.17, 'ways': 6, 'stride': 17, 'credit': 8, 'radius': 37,
            'btype': BulletTypes.R_JADE, 'color': BulletColors.PUEPLE
        }
        if augmentation:
            self.itv = transform_single(self.itv)
            self.angle, self.ad = transform_multi(self.angle, self.ad)
            self.cargs, = transform_dicts(self.cargs)

    def act(self):
        if self.cnt % round(self.itv) == 0:
            thetas = univals(0, 360, self.nshtr)
            for theta in thetas[:-1]:
                self.shooters.add(
                    D6.MovingCircle(theta=self.angle + theta,  **self.cargs)
                )
            self.angle += self.ad


# class D7(Danmaku):
#     itv = 24
#     reverse_start = 11
#     ia, ad = 10, 27
#     bend = -15
#     span = 60
#     ways = 4
#     radius = 45
#     s0, s1, n = 1.8, 2.2, 2
#     cargs = {'btype': BulletTypes.DOT, 'color': BulletColors.GREEN}
#     P2_start = 15
#     P2_itv = 90
#     P2_n = 5
#     P2_theta = 60
#     P2_bend = 160
#     P2_ad = 23
#     P2_cargs = {
#         'speed': 2.4, 'snipe': True, 'radius': 60, 'rho': 80, 'ways': 32,
#         'btype': BulletTypes.SQUAMA, 'color': BulletColors.RED
#     }
#     def __init__(self, augmentation=False):
#         super(D7, self).__init__()
#         self.angle = D7.ia
#         self.P2_theta = D7.P2_theta
#         self.P2_bend = D7.P2_bend
#         self.P2_angle = 0
#
#     def act(self):
#         if self.cnt % D7.itv == 0:
#             for s in univals(D7.s0, D7.s1, D7.n):
#                 BatchedBulletBuilder.instance.create(
#                     angle=self.angle + D7.bend, span=D7.span, ways=D7.ways, rho=D7.radius, theta=self.angle,
#                     speed=s, **D7.cargs
#                 )
#                 BatchedBulletBuilder.instance.create(
#                     angle=-(self.angle + D7.bend), span=D7.span, ways=D7.ways, rho=D7.radius, theta=-self.angle,
#                     speed=s, **D7.cargs
#                 )
#         if self.cnt % D7.itv == D7.reverse_start:
#             for s in univals(D7.s0, D7.s1, D7.n):
#                 BatchedBulletBuilder.instance.create(
#                     rho=D7.radius, theta=self.angle, angle=self.angle + D7.bend + 180, speed=s,
#                     ways=D7.ways, span=D7.span, **D7.cargs
#                 )
#                 BatchedBulletBuilder.instance.create(
#                     rho=D7.radius, theta=-self.angle, angle=-(self.angle + D7.bend + 180), speed=s,
#                     ways=D7.ways, span=D7.span, **D7.cargs
#                 )
#             self.angle += D7.ad
#         if self.cnt % D7.P2_itv == D7.P2_start:
#             BatchedBulletBuilder.instance.create(
#                 theta=self.P2_theta, bend=self.P2_bend, angle=self.P2_angle, **D7.P2_cargs
#             )
#         if self.cnt % D7.P2_itv == D7.P2_start + D7.P2_itv // 2:
#             BatchedBulletBuilder.instance.create(
#                 theta=-self.P2_theta, bend=-self.P2_bend, angle=-self.P2_angle, **D7.P2_cargs
#             )
#             self.P2_angle += D7.P2_ad
#             # self.P2_theta = -self.P2_theta
#             # self.P2_bend = -self.P2_bend
#
#         # if self.cnt % D7.P2_itv0 == 0:
#         #     self.shooters.add(
#         #         SpiningShooter(D7.P2_itv1, rho=80, theta=80, ways=5, credit=D7.P2_n, **D7.P2_cargs),
#         #         SpiningShooter(D7.P2_itv1, rho=80, theta=-80, ways=5, credit=D7.P2_n, **D7.P2_cargs)
#         #     )


