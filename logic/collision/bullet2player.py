from abc import abstractmethod
from logic.objects.player import Player
from logic.runtime import objs
# from utils.math import Vec2


class Collider:
    hr = Player.hr
    hrsqr = hr * hr

    def __init__(self):
        pass

    def get_v(self, pos, player_pos):
        if player_pos is None:
            v = objs.player.pos - pos
        else:
            v = player_pos - pos
        return v

    @abstractmethod
    def detect(self, pos, player_pos=None):
        # True: hit, Flase: nothing
        pass


class RoundCollider(Collider):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def detect(self, pos, player_pos=None):
        v = self.get_v(pos, player_pos)
        vsqr = v.x * v.x + v.y * v.y
        return vsqr < (Collider.hr + self.radius) ** 2


# class HexagenCollider(Collider):
#     def __init__(self, a, b, c):
#         super().__init__()
#         self.a, self.b, self.c, self.sin_a, self.cos_a \
#             = HexagenCollider.__resolve_config(a, b, c)
#
#         self.exclude_sqr = ((a * a + b * b) ** 0.5 + Collider.hr) ** 2   # fast exclude radius square
#         self.x_thresholds = self.__get_x_thresholds(Collider.hr)
#
#     def detect(self, pos, dire):
#         v = self.get_v(pos, dire)
#         vsqr =  v.x * v.x + v.y * v.y
#
#         if vsqr >= self.exclude_sqr:
#             return False
#         else:
#             v = v.tran_base_Xaxis(dire)
#             return v.y < self.__get_y_threshold(v.x)
#
#     def __get_x_thresholds(self, r):
#         t1 = self.c
#         t2 = self.c + r * self.sin_a
#         t3 = self.a + r * self.sin_a
#         t4 = self.a + r
#         return t1, t2, t3, t4
#
#     def __get_y_threshold(self, x):
#         if x >= self.x_thresholds[-1]:
#             return -1   # x, y have taken absolute value so a positive value can make sure no hit
#         elif x < self.x_thresholds[0]:
#             return self.b + Collider.hr
#         elif x < self.x_thresholds[1]:
#             tmp = x - self.c
#             return self.b + (Collider.hrsqr - tmp * tmp) ** 0.5
#         elif x < self.x_thresholds[2]:
#             return self.b * (self.a - x + Collider.hr * self.sin_a) / (self.a - self.c) + Collider.hr * self.cos_a
#         else:
#             tmp = x - self.a
#             return (Collider.hrsqr - tmp * tmp) ** 0.5
#
#     @staticmethod
#     def __resolve_config(a, b, c):
#         if a < 0 or b < 0:
#             raise ValueError('HexaCollider: parameters must be all positive')
#
#         real_c = c
#         if c == 'right':
#             real_c = a - b
#             if a < b:
#                 raise ValueError('HexaCollider: Right HexaCollider requires a larger than b')
#         elif c == 'rect':
#             real_c = a
#         elif c == 'rhombus':
#             real_c = 0
#
#         tmp = a - real_c
#         l = (tmp * tmp + b * b) ** 0.5
#         sin_a = b / l
#         cos_a = tmp / l
#
#         return a, b, real_c, sin_a, cos_a
