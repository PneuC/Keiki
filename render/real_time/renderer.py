from pygame.sprite import Group
from enum import Enum


class FullRenderer:
    instance = None
    size = (384, 448)

    class Layers(Enum):
        BackGround = 0
        PlayerBullet = 1
        Boss = 2
        Player = 3
        BulletGiant = 4
        BulletBig = 5
        BulletMid = 6
        BulletTiny = 7
        PlayerHitBox = 8

    def __init__(self, tar):
        self.tar = tar
        self.groups = [Group() for _ in FullRenderer.Layers]
        FullRenderer.instance = self

    def update(self):
        for group in self.groups:
            group.update()

    def draw(self):
        for group in self.groups:
            group.draw(self.tar)

    def add_sprites(self, layer, *sprites):
        self.groups[layer.value].add(*sprites)
