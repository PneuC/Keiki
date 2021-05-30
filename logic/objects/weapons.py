import pygame
from utils.math import Vec2
from logic.runtime.bullet_creation import create_player_bullet
from pygame.key import get_pressed

#
# class SimpleBullet(PlayerBullet):
#     def __init__(self, pos, velocity):
#         super().__init__(pos)
#         self.velocity = velocity
#         if self.render:
#             FullRenderer.instance.add_sprites(
#                 FullRenderer.Layers.PlayerBullet, BulletSprite(self, TexManager.inst().main)
#             )
#
#     def move(self):
#         self.pos += self.velocity
#

class Weapon:
    def __init__(self, master, btype, itv, bspeed, ports):
        self.master = master
        self.btype = btype
        self.bspeed = bspeed
        self.ports = ports
        self.itv = itv
        self.coldown = 0
        self.on = False

    def update(self):
        if not self.on:
            return
        if self.coldown > 0:
            self.coldown -= 1
        else:
            for port in self.ports:
                bpos = self.master.pos + port
                create_player_bullet(self.btype, [bpos, self.bspeed])
            self.coldown += self.itv


class Sub:
    def __init__(self, master, biases):
        self.angle = 0
        self.master = master
        self.pos = master.pos
        self.ratio = [1.0, 0.0]
        self.bias0, self.bias1 = biases

    def __getattr__(self, item):
        if item == 'render_pos':
            return self.pos.to_rcs().tuple()

    def update(self):
        status = 1 if self.master.slow_down else 0
        if self.ratio[status] < 1.0:
            self.ratio[status] += 0.2
            self.ratio[(status + 1) % 2] -= 0.2
        bias = self.ratio[0] * self.bias0 + self.ratio[1] * self.bias1
        self.pos = self.master.pos + bias
        self.angle += 3