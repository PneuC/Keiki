import json
from copy import copy
from utils.data_structure import BufferedLazyList
from utils.math import Vec2
from logic.runtime.bullet_creation import SimBatchedBulletBuilder, RuntimeBatchedBulletBuilder
from logic.runtime import objs
from logic.objects.player import Player
from logic.collision.bullet2player import Collider


class RealTimeSimulator:
    instance = None

    def __init__(self):
        # self.cnt = 0
        self.stab = []          # Absolute Cartesian coordinates of all the non-interactive bullets at each frames
        self.itab = []          # (abs)cnt, sp, rho, theta, collider
        self.running = False
        self.loaded = False
        self.T = 0
        RealTimeSimulator.instance = self

    def evaluate(self, cnt, dire_vec, repeat, trace):
        miss = 0
        t = cnt
        new_trace = copy(trace)
        # print(len(trace))
        ppos = trace[-1].cpy()
        # ppos = player_pos.cpy()
        movement = dire_vec.norm() * Player.speed_high
        Collider.hr += 2.0
        Collider.hrsqr += 12.0
        while t < min(self.T, cnt + repeat):
            if (objs.player.pos - objs.boss.pos).m < 30:
                miss = 1
            if miss == 0:
                for pos, collider in self.stab[t]:
                    if collider.detect(pos, ppos):
                        miss = 1
                for tar_cnt, sp, rho, theta, collider in self.itab[t]:
                    trace_start = cnt - len(trace)
                    index = tar_cnt - trace_start
                    if index >= 0:
                        snipe_tar = new_trace[index]
                    else:
                        snipe_tar = objs.player.trace[index]
                    # print('playercnt: %d, tarcnt: %d, statecnt: %d, tracelen: %d, index: %d' % (objs.player.cnt, tar_cnt, cnt, len(trace), index) )
                    abs_theta = (snipe_tar - sp).theta() + theta
                    pos = sp + Vec2.from_plr(rho, abs_theta)
                    if collider.detect(pos, ppos):
                        miss = 1
            ppos = (ppos + movement).bound_in(*Player.bound)
            new_trace.append(ppos.cpy())
            t += 1
        # print(ppos)
        safety = 0
        stop = False
        while t < min(cnt + repeat + 10, self.T):
            # print(len(self.static_tab[t]))
            for pos, collider in self.stab[t]:
                if collider.detect(pos, ppos):
                    stop = True
                    break
            for tar_cnt, sp, rho, theta, collider in self.itab[t]:
                trace_start = cnt - len(trace)
                index = tar_cnt - trace_start
                if index >= 0:
                    if index < len(new_trace):
                        snipe_tar = new_trace[index]
                    else:
                        snipe_tar = new_trace[-1]
                else:
                    snipe_tar = objs.player.trace[index-1]
                # print('playercnt: %d, tarcnt: %d, statecnt: %d, tracelen: %d, index: %d' % (objs.player.cnt, tar_cnt, cnt, len(trace), index) )
                abs_theta = (snipe_tar - sp).theta() + theta
                pos = sp + Vec2.from_plr(rho, abs_theta)
                if collider.detect(pos, ppos):
                    stop = True
                    break
            if stop:
                break
            safety += 1
            t += 1
        Collider.hr -= 2.0
        Collider.hrsqr -= 12.0
        # print(*(str(p) for p in trace))
        # print(*(str(p) for p in new_trace))
        # print()
        return safety, miss, new_trace

    def load(self, danmaku):
        # Load a danamku, generate static tab, interact_list
        # self.cnt = 0
        self.T = 0
        self.loaded = True

        static_bullets = BufferedLazyList()
        interactive_bullets = BufferedLazyList()

        SimBatchedBulletBuilder(self, static_bullets, interactive_bullets)
        # objs.player.no_collison = True
        self.stab = [[]]
        self.itab = [[]]
        while not danmaku.dead:
            danmaku.update()
            # print(danmaku.cnt)
            static_bullets.update()
            interactive_bullets.update()
            self.stab.append([])
            self.itab.append([])
            for b in static_bullets:
                if not b.dead:
                    self.stab[-1].append((b.pos.cpy(), b.collider))
            for b in interactive_bullets:
                if not b.dead:
                     self.itab[-1].append((b.tar_cnt - 1, b.spos.cpy(), b.rho, b.theta, b.collider))
            self.T += 1
        # objs.player.no_collison = False
        RuntimeBatchedBulletBuilder(objs.render)
        print('simulator T: ', self.T)
        # with open('stab.json', 'w') as f:
        #     json.dump([[str(b[0]) for b in frame] for frame in self.stab[:200]], f)

    def clear(self):
        self.stab.clear()
        # self.dynamic_tab.clear()
        # self.interact_list.clear()
        self.running = False
        self.loaded = False


