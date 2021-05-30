import os
from root import PRJROOT
from logic.runtime import objs
from logic.danmaku.cm_danmaku import BatchedCMDanmaku
from utils.math import Vec2


class Boss:
    spell_interval = 90
    spell_hp = 600

    def __init__(self, pos=None):
        self.cnt = 0
        if pos is None:
            self.pos = Vec2(0, 72)
        else:
            self.pos = pos

        self.spells = []
        for root, _, files in os.walk(PRJROOT + 'danmakus'):
            for fname in files:
                self.spells.append(PRJROOT + 'danmakus/' + fname)

        self.scid = 0
        self.spell = None
        self.wait = 0
        self.hp = self.spell_hp
        self.move_itv = 0

    def update(self):
        if self.scid >= len(self.spells):
            return
        if self.spell is None:
            objs.bullets.update()
            self.load(self.spells[self.scid])
            self.wait += Boss.spell_interval
            self.hp = self.spell_hp
            return
        if self.spell.dead:
            self.spell = None
            self.scid += 1
            return
        if self.wait > 0:
            self.wait -= 1
            if self.wait == 60:
                for b in objs.bullets:
                    b.dead = True
        else:
            self.spell.update()
            if self.hp <= 0:
                self.spell.dead = True
        if (objs.player.pos - self.pos).m < 30:
            objs.player.miss()
        self.cnt += 1

    def load(self, danmaku):
        if type(danmaku) == str:
            self.spell = BatchedCMDanmaku()
            self.spell.load_file(danmaku)
        elif type(danmaku) == type:
            self.spell = danmaku()
        else:
            self.spell = danmaku

    def __getattr__(self, item):
        if item == 'pos':
            return self.pos
        elif item == 'render_pos':
            return self.pos.to_rcs().tuple()

