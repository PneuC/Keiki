from abc import abstractmethod
from math import sin, tan, radians
from logic.runtime import objs
from logic.objects.bullet_enums import *
from logic.danmaku.basis import PlayerPosKeeper
from logic.collision.bullet2player import RoundCollider
from logic.objects.bullets import Bullet, StaticSimBullet, InteractiveSimBullet, PlayerBullet
from render.real_time.common_sprites import CommonSprite
from render.real_time.renderer import FullRenderer
from utils.assets_manage import TexManager
from utils.math import Vec2, bound, linear_mapping


def create_player_bullet(btype, args):
    bullet = PlayerBullet(*args)
    objs.others.add(bullet)
    if not objs.render:
        return
    if btype == 'sub0':
        btex = TexManager.inst().sub0
    elif btype == 'sub1':
        btex = TexManager.inst().sub1
    else:
        btex = TexManager.inst().main
    FullRenderer.instance.add_sprites(
        FullRenderer.Layers.PlayerBullet, CommonSprite(bullet, btex)
    )

def circle_cfgs(pos, angle, bend, radius, ways):
    angle_diff = 360 / ways
    res = [
        (pos + Vec2.from_plr(radius, angle + angle_diff * i), angle + angle_diff * i + bend)
        for i in range(ways)
    ]
    return res

def sector_cfgs(span, pos, angle, bend, radius, ways):
    if ways == 1:
        return [(pos + Vec2.from_plr(radius, angle), angle + bend)]
    elif ways > 1:
        lower_angle = angle - (span % 360) / 2
        angle_diff = span / (ways - 1)
        res = [
            (pos + Vec2.from_plr(radius, angle), lower_angle + angle_diff * i + bend)
            for i in range(ways)
        ]
        return res


def string_cfgs(length, interior_angle, dire=0, pos=None, radius=0.0, ways=1):
    if pos is None:
        pos = Vec2(0, 0)
    pos = pos + Vec2.from_plr(radius, dire)
    if ways == 1:
        return [(pos, dire)]
    norm = Vec2.from_plr(1, dire)
    interior_angle %= 360
    if interior_angle <= 1:
        d = length / (ways - 1)
        v = Vec2(-norm.y, norm.x).norm()
        endpoint = pos - v * (length / 2)
        res = [
            (endpoint + v * (i * d), dire)
            for i in range(ways)
        ]
        return res
    else:
        l = 0.5 * length
        alpha = interior_angle / 2
        R = l / sin(radians(alpha))
        O = pos - Vec2.from_plr(l / tan(radians(alpha)), dire)
        lower_angle = dire - alpha
        ad = interior_angle / (ways - 1)

        res = []
        for i in range(ways):
            a = lower_angle + i * ad
            config = (O + Vec2.from_plr(R, a), a)
            res.append(config)
        return res


class BulletBuilder:
    class Coords(Enum):
        ABSOLUTE = 0
        BOSS_RELATE = 1
        PLAYER_RELATE = 2

    instance = None
    _default_kwargs = {
        'pos': None, 'angle': 0.0, 'burst': 0.0, 'frac': 0.1, 'coord': Coords.BOSS_RELATE,
        'snipe': False, 'btype': BulletTypes.S_JADE, 'color': BulletColors.WHITE
    }

    def __init__(self):
        BulletBuilder.instance = self

    @abstractmethod
    def create(self, **kwargs):
        pass


class RuntimeBulletBuilder(BulletBuilder):
    collider_mapping = {
        BulletTypes.DOT: RoundCollider(2.4),

        BulletTypes.DROP:  RoundCollider(3.5),
        BulletTypes.STAR: RoundCollider(4.0),
        BulletTypes.BULLET:  RoundCollider(3.0),
        BulletTypes.OFUDA: RoundCollider(3.5),
        BulletTypes.KUNAI: RoundCollider(2.5),
        BulletTypes.CORN:  RoundCollider(3.0),
        BulletTypes.S_JADE: RoundCollider(4.0),
        BulletTypes.R_JADE:  RoundCollider(4.0),
        BulletTypes.SQUAMA:  RoundCollider(4.5),

        BulletTypes.DAGGER: RoundCollider(5.0),
        BulletTypes.M_JADE: RoundCollider(8.5),
        BulletTypes.B_STAR: RoundCollider(6.0),
        BulletTypes.ELLIPSE: RoundCollider(5.5),

        BulletTypes.G_JADE: RoundCollider(15.0)
    }
    spin_set = {BulletTypes.STAR, BulletTypes.B_STAR}
    round_set = {
        BulletTypes.S_JADE, BulletTypes.M_JADE, BulletTypes.G_JADE, BulletTypes.R_JADE, BulletTypes.DOT
    }

    def __init__(self, render=True):
        # 导入boss模块会导致循环导入，所以在创建时把boss的引用放进对象的属性里
        # self.boss = boss
        # self.player = player
        self.render = render
        super().__init__()

    def create(self, **kwargs):
        """
        coord: Coords = BOSS_RELATE, pos: Vec2, snipe: [bool | PlayerPosKeeper] = None, angle: float = 0.0
        init_speed: float = None, frac float = 0.0, btype: BulletTypes = S_JADE,
        color: BulletColors = SILIVER
        TODO: uniform_ab, gaussian_ab, steer_force, acceleration, delay
        """
        kwparas = self._default_kwargs.copy()
        kwparas.update(kwargs)
        if kwparas['pos'] is None:
            kwparas['pos'] = Vec2(0, 0)
        if kwparas['coord'] == self.Coords.BOSS_RELATE:
            kwparas['pos'] += objs.boss.pos
        elif kwparas['coord'] == self.Coords.PLAYER_RELATE:
            kwparas['pos'] += objs.player.pos

        if type(kwparas['snipe']) == bool:
            if kwparas['snipe']:
                v = objs.player.pos - kwparas['pos']
                kwparas['angle'] += v.theta()
        elif type(kwparas['snipe']) == Vec2:
            v = kwparas['snipe'] - kwparas['pos']
            kwparas['angle'] += v.theta()
        elif not kwparas['snipe'] is None :
            v = kwparas['snipe'].val - kwparas['pos']
            kwparas['angle'] += v.theta()

        kwparas['power'] = kwparas['speed'] * kwparas['frac']
        kwparas['velocity'] = Vec2.from_plr(kwparas['burst'] + kwparas['speed'], kwparas['angle'])
        btype, color = kwparas['btype'], kwparas['color']
        if btype in self.spin_set:
            kwparas['spin'] = 2.4
        else:
            kwparas['spin'] = 0.0
        kwparas['out_range'] = get_out_range(btype)
        kwparas['weight'] = get_weight(btype)
        kwparas['collider'] = self.collider_mapping[btype]

        bullet = Bullet(**kwparas)
        objs.bullets.add(bullet)
        if self.render:
            sprite_angle = 0.0 if btype in self.round_set else kwparas['angle']
            spinable = btype in self.spin_set
            FullRenderer.instance.add_sprites(
                get_render_layer(btype),
                CommonSprite(
                    bullet, TexManager.inst().get_bullet_tex(btype, color),
                    sprite_angle, spinable
                )
            )


class EncodingBulletBuilder(BulletBuilder):
    """
        0: x, 1: y, 2: angle, 3: speed, 4: frac,
        5: init_speed, 6: snipe, 7: type, 8: color, 9: delay
    """
    def __init__(self, one_hot=False):
        super().__init__()
        self.buffer = []
        self.cnt = 0
        self.bcnt = 0
        self.one_hot = one_hot

    def update(self):
        self.cnt += 1

    def create(self, **kwargs):
        self.bcnt += 1
        item = [0.0] * 10
        kwparas = self._default_kwargs.copy()
        kwparas.update(kwargs)
        if kwparas['pos'] is None:
            kwparas['pos'] = Vec2(0.0, 0.0)
        if type(kwparas['pos']) == Vec2:
            item[0] = (kwparas['pos'].x + 192) / 384
            item[1] = (kwparas['pos'].y + 224) / 448
        item[2] = (kwparas['angle'] % 360 + 20) / 400
        item[3] = bound(kwparas['speed'], 0.0, 6.0) / 6.0

        item[4] = bound(kwparas['frac'], 0.0, 4.0) / 4.0
        item[5] = bound(kwparas['burst'] + 0.2, -0.2, 19.8) / 20.0
        if not kwparas['snipe']:
            item[6] = -1
        elif kwparas['snipe'] is True:
            item[6] = 0
        else:
            item[6] = kwparas['snipe'].cnt
        item[6] = bound(item[6] + 2.0, -2.0, 98.0) / 100.0
        item[7] = kwparas['btype'].value
        item[8] = kwparas['color'].value
        item[9] = bound(self.cnt + 5.0, 0.0, 100.0) / 100.0
        self.cnt = 0
        self.buffer.append(item)

    def collect(self):
        data = self.buffer
        del self.buffer
        self.buffer = []
        self.bcnt = 0
        return data


class BatchedBulletBuilder:
    instance = None
    _default_kwargs = {
        'rho': 0.0, 'theta': 0.0, 'angle': 0.0, 'speed': 1.0, 'burst': 0.0, 'decay': 0.5, 'radius': 0.0,
        'btype': BulletTypes.S_JADE, 'color': BulletColors.WHITE, 'strlen': 0.0,
        'bend': 0.0, 'ways': 1, 'span': 360., 'snipe': False
    }

    def __init__(self):
        BatchedBulletBuilder.instance = self

    @abstractmethod
    def create(self, **kwargs):
        pass


class RuntimeBatchedBulletBuilder(BatchedBulletBuilder):
    def __init__(self, render=True):
        super(RuntimeBatchedBulletBuilder, self).__init__()
        self.render = render

    @abstractmethod
    def create(self, **kwargs):
        params = BatchedBulletBuilder._default_kwargs.copy()
        params.update(kwargs)
        if params['rho'] < 10.:
            params['rho'] = 0.
        if params['radius'] < 10.:
            params['radius'] = 0.
        if params['burst'] < 0.5:
            params['burst'] = 0.
        if params['bend'] < 10.:
            params['bend'] = 0.
        pos = Vec2.from_plr(params['rho'], params['theta']) + objs.boss.pos
        if type(params['ways']) == float:
            params['ways'] = round(params['ways'])
        if params['ways'] < 1:
            params['ways'] = 1
        if params['strlen'] >= 10.0:
            cfgs = string_cfgs(params['strlen'], params['span'], params['angle'], pos, params['radius'], params['ways'])
        elif not 0.1 < params['span'] <= 340.:
            cfgs = circle_cfgs(pos, params['angle'], params['bend'], params['radius'], params['ways'])
        else:
            cfgs = sector_cfgs(params['span'], pos, params['angle'], params['bend'], params['radius'], params['ways'])
        # process common parameters

        btype, color = params['btype'], params['color']
        if btype in RuntimeBulletBuilder.spin_set:
            spin = 2.4
        else:
            spin = 0.0
        out_range = get_out_range(btype)
        weight = get_weight(btype)
        collider = RuntimeBulletBuilder.collider_mapping[btype]

        for pos, agl in cfgs:
            agl_bias = 0.0
            snipe = params['snipe']
            if type(snipe) == bool:
                if snipe:
                    v = objs.player.pos - pos
                    agl_bias = v.theta()
            elif type(snipe) == int and snipe >= 0:
                try:
                    v = objs.player.trace[-1 - params['snipe']] - pos
                except IndexError:
                    v = objs.player.trace[-1] - pos
                agl_bias = v.theta()
            elif type(snipe) == PlayerPosKeeper:
                v = params['snipe'].val - pos
                agl_bias = v.theta()
            agl += agl_bias
            init_spd = params['speed'] + params['burst']
            pw = params['decay'] * params['speed']
            bullet = Bullet(
                pos=pos, angle=agl, speed=init_spd, frac=params['decay'], power=pw, spin=spin,
                out_range=out_range, collider=collider, weight=weight
            )
            objs.bullets.add(bullet)
            if self.render:
                sprite_angle = 0.0 if btype in RuntimeBulletBuilder.round_set else agl
                spinable = btype in RuntimeBulletBuilder.spin_set
                FullRenderer.instance.add_sprites(
                    get_render_layer(btype),
                    CommonSprite(
                        bullet, TexManager.inst().get_bullet_tex(btype, color),
                        sprite_angle, spinable
                    )
                )


class SimBatchedBulletBuilder(BatchedBulletBuilder):
    __corners = (Vec2(-192, 224), Vec2(192, 224), Vec2(192, -224), Vec2(-192, -224))

    def __init__(self, simulator, slist, ilist):
        super(SimBatchedBulletBuilder, self).__init__()
        self.simulator = simulator
        self.slist = slist
        self.ilist = ilist

    def create(self, **kwargs):
        # Additional parameter: cnt
        params = BatchedBulletBuilder._default_kwargs.copy()
        params.update(kwargs)
        if params['rho'] < 10.:
            params['rho'] = 0.
        if params['radius'] < 10.:
            params['radius'] = 0.
        if params['burst'] < 0.5:
            params['burst'] = 0.
        if params['bend'] < 10.:
            params['bend'] = 0.
        pos = Vec2.from_plr(params['rho'], params['theta']) + objs.boss.pos
        if type(params['ways']) == float:
            params['ways'] = round(params['ways'])
        if params['strlen'] >= 10.0:
            cfgs = string_cfgs(params['strlen'], params['span'], params['angle'], pos, params['radius'], params['ways'])
        elif not 0.1 < params['span'] <= 340.:
            cfgs = circle_cfgs(pos, params['angle'], params['bend'], params['radius'], params['ways'])
        else:
            cfgs = sector_cfgs(params['span'], pos, params['angle'], params['bend'], params['radius'], params['ways'])
        # process common parameters

        btype, color = params['btype'], params['color']
        if btype in RuntimeBulletBuilder.spin_set:
            spin = 2.4
        else:
            spin = 0.0
        out_range = get_out_range(btype)
        collider = RuntimeBulletBuilder.collider_mapping[btype]

        for pos, agl in cfgs:
            # agl_bias = 0.0
            interact = False
            tar_cnt = self.simulator.T
            if type(params['snipe']) == bool:
                if params['snipe']:
                    interact = True
            elif type(params['snipe']) == int:
                tar_cnt -= params['snipe']
                interact = True
            elif not params['snipe'] is None:
                tar_cnt -= params['snipe'].cnt
                interact = True

            init_spd = params['speed'] + params['burst']
            pw = params['decay'] * params['speed']
            if interact:
                max_dis = max((pos-corner).m for corner in SimBatchedBulletBuilder.__corners)
                out_radius = out_range[0][1] - 192
                bullet = InteractiveSimBullet(
                    pos=pos, angle=agl, speed=init_spd, frac=params['decay'], power=pw, spin=spin,
                    out_rho=max_dis + out_radius, collider=collider, tar_cnt=tar_cnt
                )
                self.ilist.add(bullet)
                continue

            # agl += agl_bias
            # velo = Vec2.from_plr(init_spd, agl)
            bullet = StaticSimBullet(
                pos=pos, angle=agl, speed=init_spd, frac=params['decay'], power=pw, spin=spin,
                out_range=out_range, collider=collider
            )
            self.slist.add(bullet)


class EncodingBatchedBulletBuilder(BatchedBulletBuilder):
    encoding_scheme = (
        {'key': 'btype', 'dtype': BulletTypes, 'range': (0, 15)},
        {'key': 'color', 'dtype': BulletColors, 'range': (0, 6)},
        {'key': 'rho', 'dtype': float, 'range': (0., 250.)},
        {'key': 'theta', 'dtype': float, 'range': (0., 360.)},
        {'key': 'angle', 'dtype': float, 'range': (0., 360.)},
        {'key': 'speed', 'dtype': float, 'range': (0.6, 7.)},
        {'key': 'burst', 'dtype': float, 'range': (0., 20.)},
        {'key': 'decay', 'dtype': float, 'range': (0.1, 5.)},
        {'key': 'radius', 'dtype': float, 'range': (0., 150.)},
        {'key': 'strlen', 'dtype': float, 'range': (0., 100.)},
        {'key': 'bend', 'dtype': float, 'range': (0., 360.)},
        {'key': 'ways', 'dtype': int, 'range': (0.5, 60.)},
        {'key': 'span', 'dtype': float, 'range': (0., 360.)},
        {'key': 'snipe', 'dtype': int, 'range': (0., 100.)},
    )
    delay_cfg = {'range': (0, 150), 'bottom': 0.2}
    vrange = (0.2, 0.8)

    def __init__(self):
        super(EncodingBatchedBulletBuilder, self).__init__()
        self.buffer = []
        self.cnt = 0
        self.api_cnt = 0

    def update(self):
        self.cnt += 1

    def create(self, **kwargs):
        params = BatchedBulletBuilder._default_kwargs.copy()
        params.update(kwargs)
        scheme = EncodingBatchedBulletBuilder.encoding_scheme
        item = [0.0] * (len(scheme) + 1)
        for i in range(len(scheme)):
            cfg = scheme[i]
            if cfg['key'] == 'snipe': # < 0.15: False, 0.15 ~ 0.25: True, >= 0.25 previous position
                param = params['snipe']
                if type(param) == bool:
                    item[i] = 0.1 if not param else 0.2
                elif type(param) == int:
                    if param > 0:
                        item[i] = linear_mapping(cfg['range'], (0.25, self.vrange[1]), param)
                    else:
                        item[i] = 0.1 if param < 0 else 0.2
                elif type(param) == PlayerPosKeeper:
                    orival = params['snipe'].cnt
                    item[i] = linear_mapping(cfg['range'], (0.25, self.vrange[1]), orival)
                else:
                    raise TypeError('Bad Type for snipe: %s' % type(param))
                continue
            if cfg['key'] in {'theta', 'angle', 'span', 'bend'}:
                params[cfg['key']] = params[cfg['key']] % 360.0
                if cfg['key'] == 'span' and params[cfg['key']] == 0.:
                    params[cfg['key']] = 360.
            orival = params[cfg['key']]
            if type(orival) in {float, int}:
                orival = params[cfg['key']]
                if not cfg['range'][0] <= orival <= cfg['range'][1]:
                    fmt = '%s should in range (%.2f, %.2f) but got %.2f'
                    # if type(orival) == float:
                    #     fmt = '%s should in range (%.2f, %.2f) but got %.2f'
                    # else:
                    #     fmt = '%s should in range (%d, %d) but got %d'
                    info = fmt % (cfg['key'], cfg['range'][0], cfg['range'][1], orival)
                    raise ValueError(info)
                item[i] = linear_mapping(cfg['range'], self.vrange, orival)
            else: # Is Enum
                item[i] = orival.value
        if self.cnt > self.delay_cfg['range'][1]:
            raise ValueError('Delay shouldn\'t exceed %d' % self.delay_cfg['range'][1])
        if self.cnt == 0:
            item[-1] = self.delay_cfg['bottom'] / 2
        else:
            vrange = (self.delay_cfg['bottom'], self.vrange[1])
            item[-1] = linear_mapping(self.delay_cfg['range'], vrange, self.cnt)
        self.buffer.append(item)
        self.cnt = 0
        self.api_cnt += 1

    def collect(self):
        data = self.buffer
        del self.buffer
        self.buffer = []
        self.api_cnt = 0
        return data


# class EncodingBatchedBulletBuilder2(BatchedBulletBuilder):
#     scheme = (
#         {'key': 'btype', 'dtype': BulletTypes, 'range': (0, 15)},
#         {'key': 'color', 'dtype': BulletColors, 'range': (0, 6)},
#         {'key': 'rho', 'dtype': float, 'range': (0., 200.)},
#         {'key': 'theta', 'dtype': float, 'range': (0., 360.)},
#         {'key': 'angle', 'dtype': float, 'range': (0., 360.)},
#         {'key': 'speed', 'dtype': float, 'range': (0.6, 6.)},
#         {'key': 'burst', 'dtype': float, 'range': (0., 15.)},
#         {'key': 'decay', 'dtype': float, 'range': (0.1, 4.)},
#         {'key': 'radius', 'dtype': float, 'range': (0., 100.)},
#         {'key': 'strlen', 'dtype': float, 'range': (0., 100.)},
#         {'key': 'bend', 'dtype': float, 'range': (0., 360.)},
#         {'key': 'ways', 'dtype': int, 'range': (1, 50)},
#         {'key': 'span', 'dtype': float, 'range': (0, 360)},
#         {'key': 'snipe', 'dtype': int, 'range': (-1, 100)},
#     )
#     """
#         Tokens: btype(15), color(6)
#         Fixed Args: rho, theta, angle, speed, burst, decay, ways, snipe, bend
#     """
#     def __init__(self):
#         super(EncodingBatchedBulletBuilder2, self).__init__()
#
#     def create(self, **kwargs):
#         pass
