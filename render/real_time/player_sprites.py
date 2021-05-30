from pygame import Rect
from pygame.sprite import Sprite
from utils.graphics import BaseSprite
from utils.assets_manage import TexManager


class PlayerBody(Sprite):
    __static_itv = 6
    __dynamic_itv = 2

    def __init__(self, master):
        super().__init__()
        self.master = master
        self.i, self.j = 0, 0
        self.sheet = TexManager.inst().player_body
        self.cnt = 0
        self.rect = Rect(0, 0, 32, 48)
        self.rect.center = master.render_pos

    def update(self):
        dire_vec = self.master.dire_vec
        if not dire_vec.x:
            if self.i != 0:
                self.i = 0
                self.cnt = 0
            elif self.cnt % PlayerBody.__static_itv == 0:
                self.j += 1
                self.j %= 8
            self.cnt += 1
        else:
            row = 1 if dire_vec.x < 0.0 else 2
            if self.i != row:
                self.cnt = 0
                self.i = row
            if self.j < 4 and self.cnt % PlayerBody.__dynamic_itv == 0:
                self.j += 1
            elif self.cnt % PlayerBody.__static_itv == 0:
                self.j += 1
                self.j %= 4
                self.j += 4
            self.cnt += 1
        self.image = self.sheet[self.i][self.j]
        self.rect.center = self.master.render_pos


class PlayerHitBox(BaseSprite):
    __a_speed = 2.75

    def __init__(self, master, odevity):
        odevity %= 2
        tex = TexManager.inst().hitboxes[odevity]
        super().__init__(tex)
        self.master = master
        self.cnt = 0
        self.angle = 0
        self.spin = 1 if odevity else -1
        self.set_visible(True)

    def update(self):
        if not self.master.slow_down:
            self.cnt = 0
            self.angle = 0
            self.set_visible(False)
        else:
            self.set_visible(True)
            if self.cnt <= 6:
                self.set_size([64 + 4 * self.cnt] * 2)
            elif self.cnt <= 12:
                self.set_size([112 - 4 * self.cnt] * 2)
            else:
                self.set_angle(self.angle)
            self.rect.center = self.master.render_pos
            self.cnt += 1
            self.angle += PlayerHitBox.__a_speed * self.spin
        self.set_angle(self.angle)
        self.rect.center = self.master.render_pos
        self.cnt += 1
        self.angle += PlayerHitBox.__a_speed * self.spin
