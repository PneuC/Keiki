from logic.objects.bullet_enums import BulletTypes, BulletColors
from logic.runtime.bullet_creation import BatchedBulletBuilder
from random import randrange, uniform
from logic.danmaku.basis import Danmaku
from logic.danmaku.common_shooters import SpiningShooter


class TestDanmaku(Danmaku):
    def __init__(self):
        super(TestDanmaku, self).__init__()

    def act(self):
        if self.cnt % 60 == 0:
            BatchedBulletBuilder.instance.create(
                ways=8, btype=BulletTypes.CORN, color=BulletColors.RED,
                speed=1.5, burst=10.0, decay=0.45, strlen=72, snipe=True
            )
