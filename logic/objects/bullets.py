from utils.math import Vec2
from logic.runtime import objs


class Bullet:
    def __init__(self, **kwargs):
        self.dead = False
        self.start_pos = kwargs['pos'].cpy()
        self.pos = kwargs['pos']
        self.theta = kwargs['angle']
        self.angle = kwargs['angle']
        # self.velocity = kwargs['velocity']
        self.speed = kwargs['speed']
        self.frac = kwargs['frac']
        self.power = kwargs['power']
        self.spin = kwargs['spin']
        self.out_range = kwargs['out_range']
        self.collider = kwargs['collider']
        self.weight = kwargs['weight']
        self.rho = 0.

    def __getattr__(self, item):
        if item == 'render_pos':
            return self.pos.to_rcs().tuple()

    def update(self):
        if self.dead:
            return
        if not objs.player.no_collison:
            collision = self.collider.detect(self.pos)
            if collision:
                self.dead = True
                objs.player.miss()
                return
        self.angle += self.spin
        self.move()
        if self.pos.is_out(*self.out_range):
            self.dead = True

    def move(self):
        self.rho += self.speed
        self.pos = self.start_pos + Vec2.from_plr(self.rho, self.theta)
        if self.power > 0:
            rt = 1.0
            while rt > 0.0:
                F = self.power / self.speed
                a = F - self.frac
                dt = min(rt, self.speed / (abs(a) + 1e-5) / 3)
                self.speed += a * dt
                rt -= dt


class StaticSimBullet:
    def __init__(self, **kwargs):
        self.dead = False
        self.start_pos = kwargs['pos']
        self.pos = kwargs['pos']

        self.angle = kwargs['angle']
        self.speed = kwargs['speed']
        self.rho = 0.
        # self.velocity = kwargs['velocity']
        self.frac = kwargs['frac']
        self.power = kwargs['power']
        self.out_range = kwargs['out_range']
        self.collider = kwargs['collider']

    def update(self):
        if self.dead:
            return
        self.move()
        if self.pos.is_out(*self.out_range):
            self.dead = True

    def move(self):
        self.rho += self.speed
        self.pos = self.start_pos + Vec2.from_plr(self.rho, self.angle)
        if self.power > 0:
            rt = 1.0
            while rt > 0.0:
                F = self.power / self.speed
                a = F - self.frac
                dt = min(rt, self.speed / (abs(a) + 1e-5) / 3)
                self.speed += a * dt
                rt -= dt


class InteractiveSimBullet:
    def __init__(self, **kwargs):
        self.dead = False
        self.rho = 0
        self.theta = kwargs['angle']
        self.speed = kwargs['speed']
        self.frac = kwargs['frac']
        self.power = kwargs['power']
        self.spin = kwargs['spin']
        self.out_rho = kwargs['out_rho']

        self.tar_cnt = kwargs['tar_cnt']
        self.spos = kwargs['pos'].cpy()
        self.collider = kwargs['collider']

    def update(self):
        if self.dead:
            return
        self.theta += self.spin
        self.move()
        if self.rho > self.out_rho:
            self.dead = True

    def move(self):
        self.rho += self.speed
        if self.power > 0:
            rt = 1.0
            while rt > 0.0:
                F = self.power / self.speed
                a = F - self.frac
                dt = min(rt, self.speed / (abs(a) + 1e-5) / 3)
                self.speed += a * dt
                rt -= dt

    @staticmethod
    def abs_pos(rho, theta, spos, ppos):
        zdire = (ppos - spos).theta()
        return Vec2.from_plr(rho, zdire + theta)


class PlayerBullet:
    render = True

    def __init__(self, pos, speed):
        self.pos = pos
        self.velocity = Vec2(0, speed)
        self.dead = False

    def __getattr__(self, item):
        if item == 'render_pos':
            return self.pos.to_rcs().tuple()

    def move(self):
        self.pos += self.velocity

    def update(self):
        if self.dead:
            return
        v = self.pos - objs.boss.pos
        if -40 < v.x < 40 and -48 < v.y < 40:
            self.dead = True
            if objs.boss.wait <= 0:
                objs.boss.hp -= 1
        else:
            self.move()
            if self.pos.is_out((-200, 200), (-224, 256)):
                self.dead = True

#
#
# if __name__ == '__main__':
#     angles = [0.0] * 30
#     for i in range(1, 30):
#         angles[i] = angles[i-1] + 12
#     static_bullets = [
#         StaticSimBullet(
#             pos=Vec2(0, 0), angle=a, velocity=Vec2.from_plr(5, a), frac=0.5, power=1.5,
#             spin=0, out_range=[(-192, 192), (-224, 224)], collider=None
#         ) for a in angles
#     ]
#     iteract_bullets = [
#         InteractiveSimBullet(
#             pos=Vec2(0, 0), angle=a, speed=5, frac=0.5, power=1.5, cnt=0,
#             spin=0, collider=None
#         ) for a in angles
#     ]
#     for _ in range(50):
#         for b in static_bullets:
#             b.update()
#         for b in iteract_bullets:
#             b.update()
#
#     # diff = Vec2.from_plr(bullets[1].pos.m, 0) - bullets[0].pos
#     # print(diff.x, diff.y)
#     ppos = Vec2.from_plr(180.0, 12.0)
#     for i in range(1, 30):
#         ib = iteract_bullets[i-1]
#         sb = static_bullets[i]
#         interact_pos = Vec2.from_plr(ib.rho, (ppos - ib.starting_pos).theta() + ib.theta)
#         diff = interact_pos - sb.pos
#         print(diff.x, diff.y)

