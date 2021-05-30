from utils.graphics import BaseSprite


class CommonSprite(BaseSprite):
    def __init__(self, master, tex, angle=0, spinable=False):
        super().__init__(tex)
        self.master = master
        self.spinable = spinable
        self.set_angle(angle)
        self.rect.center = self.master.render_pos

    def update(self):
        if self.spinable:
            self.set_angle(self.master.angle)

        self.rect.center = self.master.render_pos
        if self.master.dead:
            self.kill()

