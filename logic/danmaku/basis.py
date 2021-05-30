from abc import abstractmethod, ABC
from utils.data_structure import BufferedLazyList
from logic.runtime import objs


class PlayerPosKeeper:
    def __init__(self):
        if not objs.player is None:
            self.val = objs.player.pos
        self.cnt = 0

    def update(self):
        self.cnt += 1


class Shooter:
    def __init__(self, delay=0):
        self.cnt = 0
        self.dead = False
        if type(delay) == float:
            delay = round(delay)
        self.delay = delay
        self.snipe = None

    @abstractmethod
    def enabled(self):
        pass

    def finish(self):
        return False

    @abstractmethod
    def shot(self):
        pass

    def update(self):
        if self.finish():
            self.dead = True
        if self.dead:
            return
        if self.delay > 0:
            self.delay -= 1
            return
        if self.enabled():
            self.shot()
        if type(self.snipe) == PlayerPosKeeper:
            self.snipe.update()
        self.cnt += 1


class PeriodicShooter(Shooter, ABC):
    def __init__(self, interval, delay=0):
        super(PeriodicShooter, self).__init__(delay)
        if type(interval) == float:
            interval = round(interval)
        self.interval = interval

    def enabled(self):
        if self.delay > 0:
            return False
        return self.cnt % self.interval == 0


class Danmaku:
    def __init__(self, timeout=1500, move_config=(240, 240)):
        self.cnt = 0
        self.timeout = timeout
        self.shooters = BufferedLazyList()
        self.dead = False
        self.move_itv = move_config[0]
        self.move_start = move_config[1]

    def update(self):
        if self.dead:
            return
        self.dead = self.finish()
        if self.dead:
            return
        # t = self.cnt - self.move_start
        # if self.move_itv > 0 and t >= 0 and t % self.move_itv == 0:
        #     try:
        #         objs.boss.start_move_event()
        #     except AttributeError:
        #         pass
        self.act()
        self.shooters.update(lambda x:x.dead)
        self.cnt += 1

    # @abstractmethod
    def act(self):
        pass

    def finish(self):
        return 0 < self.timeout < self.cnt

    def post_event(self):
        pass

#
# class DecodingDanmaku(Danmaku):
#     def __init__(self, data):
#         super().__init__()
#         self.data = data
#         self.player_tracker = []
#
#     def act(self):
#         pass
