from pygame import image, Rect
from utils.graphics import BaseSprite
from root import PRJROOT


class Background(BaseSprite):
    __instance = None
    __image_path = PRJROOT + 'assets/background/bg.png'
    __a_speed = 0.24

    def __init__(self):
        super().__init__(image.load(Background.__image_path))
        # self.dire = 0
        self.rect = Rect(0, 0, 384, 448)
        self.center = self.rect.center

    # def update(self):
    #     self.set_angle(self.dire)
    #     self.rect.center = self.center
    #     self.dire += self.__a_speed
