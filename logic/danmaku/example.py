from logic.objects.bullet_enums import *
from logic.danmaku.common_shooters import *
from logic.runtime.bullet_creation import BatchedBulletBuilder, BulletBuilder
from data.code.augmentation import *
from random import shuffle, uniform


class ExampleDanmaku(Danmaku):
    def __init__(self):
        super().__init__()
        # parameters
        self.interval1 = 8
        self.ways1 = 6
        self.speed1 = 1.5
        self.angular_speed = 0.65

        self.interval2 = 60
        self.ways2 = 48
        self.speed2 = 1.8
        self.decay = 1.0
        self.burst = 5.0

    def act(self):
        if self.cnt % self.interval1 == 0:
            # self.cnt is inherited from superclass
            BatchedBulletBuilder.instance.create(
                btype=BulletTypes.CORN, color=BulletColors.YELLOW, ways=self.ways1,
                speed=self.speed1, angle=self.cnt * self.angular_speed
            )
            BatchedBulletBuilder.instance.create(
                btype=BulletTypes.CORN, color=BulletColors.YELLOW, ways=self.ways1,
                speed=self.speed1, angle=-self.cnt * self.angular_speed
            )
        if self.cnt % self.interval2 == 20:
            BatchedBulletBuilder.instance.create(
                btype=BulletTypes.CORN, color=BulletColors.RED, ways=self.ways2,
                speed=self.speed2, burst=self.burst, decay=self.decay, snipe=True
            )
