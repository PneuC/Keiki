import pygame
from utils.math import Vec2


class MissEvent:
    def __init__(self, master):
        master.protect += 180
        master.pos = Vec2(0, -224)
        self.cnt = 0
        self.dead = False

    def update(self, master):
        if master.pos.y < -172.0:
            master.pos += Vec2(0, 1.0)
        else:
            self.dead = True

#
# class PlayerBullet:
#     render = True
#
#     def __init__(self, pos):
#         self.pos = pos
#         self.dead = False
#
#     def __getattr__(self, item):
#         if item == 'render_pos':
#             return self.pos.to_rcs().tuple()
#
#     @abstractmethod
#     def move(self):
#         pass
#
#     def update(self):
#         if self.dead:
#             return
#         v = self.pos - objs.boss.pos
#         if -40 < v.x < 40 and -48 < v.y < 40:
#             self.dead = True
#             if objs.boss.wait <= 0:
#                 objs.boss.hp -= 1
#         else:
#             self.move()
#             if self.pos.is_out((-200, 200), (-224, 256)):
#                 self.dead = True
#

class Player:
    hr = 2.0
    bound = [(-180, 180), (-208, 192)]
    speed_high = 3.75
    speed_low = 1.8
    init_pos = Vec2(0.0, -172.0)
    __init_life = 2
    __miss_protect = 180

    def __init__(self):
        self.__speed_high = Player.speed_high
        self.__speed_low = Player.speed_low
        self.pos = Player.init_pos.cpy()
        self.dire_vec = Vec2(0.0, 0.0)
        self.slow_down = False
        self.__life = self.__init_life
        self.status = 0
        self.protect = 0
        self.no_collison = False
        self.event = None
        self.trace = [self.pos.cpy()]
        self.cnt = 0

    def __getattr__(self, item):
        if item == 'render_pos':
            return self.pos.to_rcs().tuple()

    def update(self):
        self.trace.append(self.pos)
        if len(self.trace) > 500:
            self.trace.pop(0)
        if self.event is None:
            self.move()
            # self.shot()
        else:
            self.event.update(self)
            if self.event.dead:
                self.event = None
        if self.protect > 0:
            self.protect -= 1
        self.cnt += 1

    def move(self):
        x = self.__speed_low if self.slow_down else self.__speed_high
        self.pos += self.dire_vec.norm() * x
        self.pos.do_bound_in(*Player.bound)
        # self.dire_vec = Vec2(0.0, 0.0)

    def miss(self):
        if self.protect > 0:
            return
        self.event = MissEvent(self)
        # print('miss')

#
# class ReiMu(Player):
#     def __init__(self):
#         super().__init__(3.75, 1.8)
