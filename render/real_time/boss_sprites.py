from utils.graphics import BaseSprite
from pygame import image
from root import PRJROOT


class HoujuuBallSprites(BaseSprite):
    image_path = PRJROOT + 'assets/boss/HoujuuBall.png'
    spin_speed = 20

    def __init__(self, master):
        super().__init__(image.load(HoujuuBallSprites.image_path))
        self.master = master
        self.angle = 0
        self.rect.center = master.render_pos

    def update(self):
        self.set_angle(self.angle)
        self.rect.center = self.master.render_pos
        self.angle += HoujuuBallSprites.spin_speed

