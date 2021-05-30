from pygame.sprite import Sprite
from pygame.transform import rotate, scale
from root import VOID


class BaseSprite(Sprite):
    def __init__(self, tex):
        super().__init__()
        self.tex = tex
        self.image = tex
        self.rect = tex.get_rect()
        self.visible = True

    def set_angle(self, angle):
        if not self.visible:
            return
        self.image = rotate(self.tex, angle)
        self.rect.size = self.image.get_size()

    def set_size(self, size):
        if not self.visible:
            return
        self.image = scale(self.tex, size)
        self.rect.size = self.image.get_size()

    def set_visible(self, visible):
        if visible:
            self.image = self.tex
            self.visible = True
        else:
            self.image = VOID
            self.visible = False
