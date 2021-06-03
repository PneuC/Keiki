import os
import pygame
import numpy as np
from copy import deepcopy
from pygame import time, display, image
from logic.runtime.bullet_creation import RuntimeBulletBuilder, RuntimeBatchedBulletBuilder
from root import PRJROOT
from render.real_time.background import Background
from logic.runtime.obj_creation import *
from logic.runtime import objs
from agents.realtime_sim import RealTimeSimulator


class KeyResponser:
    useq = []
    dseq = []
    lseq = []
    rseq = []

    @staticmethod
    def response(event):
        if event.type not in (pygame.KEYUP, pygame.KEYDOWN):
            return
        if event.key == pygame.K_z:
            KeyResponser.on_z(event.type)
        elif event.key == pygame.K_UP:
            KeyResponser.on_up(event.type)
        elif event.key == pygame.K_DOWN:
            KeyResponser.on_down(event.type)
        elif event.key == pygame.K_LEFT:
            KeyResponser.on_left(event.type)
        elif event.key == pygame.K_RIGHT:
            KeyResponser.on_right(event.type)
        elif event.key == pygame.K_LSHIFT:
            KeyResponser.on_shift(event.type)
        elif event.key == pygame.K_s:
            KeyResponser.on_s(event.type)
        else:
            return

    @staticmethod
    def on_z(eventype):
        if eventype == pygame.KEYDOWN:
            for weapon in objs.weapons:
                weapon.on = True
        else:
            for weapon in objs.weapons:
                weapon.on = False

    @staticmethod
    def on_up(eventype):
        if eventype == pygame.KEYDOWN:
            KeyResponser.useq.append(0)
            objs.player.dire_vec.y = 1.0
        else:
            KeyResponser.useq.append(1)
            if objs.player.dire_vec.y == 1.0:
                objs.player.dire_vec.y = 0

    @staticmethod
    def on_down(eventype):
        if eventype == pygame.KEYDOWN:
            KeyResponser.dseq.append(0)
            objs.player.dire_vec.y = -1.0
        else:
            KeyResponser.dseq.append(1)
            if objs.player.dire_vec.y == -1.0:
                objs.player.dire_vec.y = 0

    @staticmethod
    def on_left(eventype):
        if eventype == pygame.KEYDOWN:
            KeyResponser.lseq.append(0)
            objs.player.dire_vec.x = -1.0
        else:
            KeyResponser.lseq.append(1)
            if objs.player.dire_vec.x == -1.0:
                objs.player.dire_vec.x = 0

    @staticmethod
    def on_right(eventype):
        if eventype == pygame.KEYDOWN:
            KeyResponser.rseq.append(0)
            objs.player.dire_vec.x = 1.0
        else:
            KeyResponser.rseq.append(1)
            if objs.player.dire_vec.x == 1.0:
                objs.player.dire_vec.x = 0

    @staticmethod
    def on_shift(eventype):
        if eventype == pygame.KEYDOWN:
            objs.player.slow_down = True
        else:
            objs.player.slow_down = False

    @staticmethod
    def on_s(eventype):
        if eventype == pygame.KEYUP:
            objs.boss.spell.dead = True


class Game:
    def __init__(self, render=True):
        self.render = render
        objs.render = render
        if render:
            self.screen = display.set_mode((640, 480))
            self.screen.blit(image.load(PRJROOT + 'assets/board.png'), [0, 0])
            self.renderer = FullRenderer(self.screen.subsurface(32, 16, 384, 448))
            self.renderer.add_sprites(FullRenderer.Layers.BackGround, Background())
        create_player(render)
        create_boss(render)
        self.pausing = 0
        RuntimeBatchedBulletBuilder(render)
        RuntimeBulletBuilder(render)

    def update(self):
        objs.update()
        if self.render:
            self.renderer.update()
            self.renderer.draw()

    @staticmethod
    def run(*danmakus, render=True, agent=None):
        pygame.init()
        game = Game(render)
        objs.boss.spells += danmakus
        display.flip()
        clk = time.Clock()
        screenshot_cnt = 1
        if agent:
            objs.weapons.update()
            for weapon in objs.weapons:
                weapon.on = True

        while True:
            clk.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if not game.pausing:
                        game.pausing = 2
                elif event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                    if game.pausing:
                        game.pausing -= 1
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                    if game.pausing:
                        try:
                            pygame.image.save(FullRenderer.instance.tar, PRJROOT + 'screenshots/%d.png' % screenshot_cnt)
                        except FileNotFoundError:
                            os.makedirs(PRJROOT + 'screenshots')
                        screenshot_cnt += 1
                if not agent:
                    KeyResponser.response(event)
            if game.pausing:
                continue
            if not RealTimeSimulator.instance is None:
                simulator = RealTimeSimulator.instance
                if not simulator.loaded and objs.boss.spell:
                    simulator.load(deepcopy(objs.boss.spell))
                if simulator.loaded and simulator.running and objs.boss.spell is None:
                    simulator.clear()
            if agent:
                agent.update()
                objs.player.dire_vec, objs.player.slow_down = agent.get_dvec()
            game.update()
            display.flip()


class Monitor:
    def __init__(self, winsize=16):
        self.winsize = winsize
        self.prev_wmatrix = np.zeros([384 // winsize, 448 // winsize])
        self.cur_wmatrix = np.zeros([384 // winsize, 448 // winsize])
        self.entropy = 0.
        self.max_momentum = 0.
        self.T = 0

    def update(self):
        weight_sum = 0.
        momentum_sum = 0.
        for b in objs.bullets:
            weight_sum += b.weight
            momentum_sum += b.weight * b.velocity.m
            i, j = int((b.pos.x + 192) // self.winsize), int((b.pos.y + 224) // self.winsize)
            try:
                self.cur_wmatrix[i, j] = self.cur_wmatrix[i, j] + b.weight
            except IndexError:
                pass
        if weight_sum:
            self.entropy += abs(self.cur_wmatrix - self.prev_wmatrix).sum() / weight_sum
        self.max_momentum = max(self.max_momentum, momentum_sum)
        self.cur_wmatrix, self.prev_wmatrix = self.prev_wmatrix, self.cur_wmatrix
        self.cur_wmatrix.fill(0.)
        self.T += 1

    def collect_data(self):
        return self.entropy / self.T, self.max_momentum

    def refresh(self):
        self.prev_wmatrix.fill(0.)
        self.cur_wmatrix.fill(0.)
        self.entropy = 0.


class FeatureBaseMetricEvaluator:
    def __init__(self, danmaku, w=32):
        if type(danmaku) != str:
            raise TypeError('Can only evaluate encoded danamkus from a file name')
        self.danmaku = danmaku
        objs.clear()
        objs.render = False
        create_player(False)
        create_boss(False)
        RuntimeBatchedBulletBuilder(False)
        RuntimeBulletBuilder(False)
        self.sw = w
        self.cover_map = np.zeros([384 // 8, 448 // 8, 3], int)
        self.weight_map0 = np.zeros([384 // w, 448 // w])
        self.weight_map1 = np.zeros([384 // w, 448 // w])
        self.weight_map_range = [(-192 + w / 2 + 0.1, 192 - w / 2 - 0.1), (-224 + w / 2 + 0.1, 224 - w / 2 - 0.1)]
        self.weight_map_num = 384 // w * 448 // w
        self.cover_map_num = 384 // 8 * 448 // 8

    def evluate(self):
        result = {'SF': 0, 'EFR': 0, 'MM': 0, 'DE': 0, 'C': 0}
        objs.boss.spells = [self.danmaku]
        objs.update()
        L ,T, EF = objs.boss.spell.get_features()
        result['SF'] = L / T
        result['EFR'] = EF / T
        cnt = 0
        while objs.boss.spell and cnt < 1000:
            self.update_cover_map()
            for b in objs.bullets:
                if b.pos.is_out(*self.weight_map_range):
                    continue
                i = max(int((b.pos.x + 192) / self.sw + 0.5), 0)
                j = max(int((b.pos.y + 224) / self.sw + 0.5), 0)
                self.weight_map1[i, j] += b.weight * b.speed
                result['MM'] += b.speed * b.weight
                result['DE'] += abs(self.weight_map1 - self.weight_map0).sum()
            self.weight_map0, self.weight_map1 = self.weight_map1, self.weight_map0
            objs.update()
            cnt += 1
        result['MM'] /= min(T, cnt)
        result['DE'] /= (min(T, cnt) * self.weight_map_num)
        result['C'] = self.cover_map[:, :, 0].sum() / self.cover_map_num
        return result

    def update_cover_map(self):
        for b in objs.bullets:
            if b.pos.is_out((-192, 192), (-224, 224)) or b.weight < 4.:
                continue
            i = int((b.pos.x + 192) / 8 + 0.5)
            j = int((b.pos.y + 224) / 8 + 0.5)
            l = (int(b.weight ** 0.5) - 1) // 2
            # print(l)
            if not self.cover_map[i:i, j:j, l:].sum():
                self.cover_map[max(0, i - l): i + l, max(0, j - l): j + l, 0] = 1
            try:
                self.cover_map[i, j, l] = 1
            except IndexError:
                continue


class AgentBaseMetricEvaluator:
    def __init__(self, danmaku, agent):
        if type(danmaku) != str:
            raise TypeError('Can only evaluate encoded danamkus from a file name')
        # self.danmaku = danmaku
        objs.clear()
        objs.render = False
        create_player(False)
        create_boss(False)
        RuntimeBatchedBulletBuilder(False)
        RuntimeBulletBuilder(False)
        self.agent = agent
        # print(danmaku)
        objs.boss.spells = [danmaku]

    def evaluate(self):
        objs.weapons.update()
        for weapon in objs.weapons:
            weapon.on = True
        win = False

        total = 0
        risk = 0
        while True:
            if not RealTimeSimulator.instance is None:
                simulator = RealTimeSimulator.instance
                if not simulator.loaded and objs.boss.spell:
                    simulator.load(deepcopy(objs.boss.spell))
                if simulator.loaded and simulator.running and objs.boss.spell is None:
                    simulator.clear()
            self.agent.update()
            objs.player.dire_vec, objs.player.slow_down = self.agent.get_dvec()
            objs.update()
            if objs.boss.spell is None:
                win = True
                break
            if objs.player.protect:
                break
            for b in objs.bullets:
                if (b.pos - objs.player.pos).m <= 10 * b.weight ** 0.5 + 4:
                    risk += 1
                    break
            total += 1
        result = {
            'Playable': win, 'Entropy': self.agent.entropy * self.agent.repeat / total,
            'Risk': risk / total
        }
        return result

