from enum import Enum
from render.real_time.renderer import FullRenderer


class Sizes(Enum):
    TINY = 8
    MID = 16
    BIG = 32
    GIANT = 64


class BulletTypes(Enum):
    # Tiny
    DOT = 0         # 小弹
    # Mid
    DROP = 1        # 滴弹
    STAR = 2        # 星弹
    CORN = 3        # 米弹
    BULLET = 4      # 铳弹
    OFUDA = 5       # 札弹
    KUNAI = 6       # 苦无
    S_JADE = 7      # 小玉
    R_JADE = 8       # 环玉
    SQUAMA = 9     # 鳞弹
    # Big
    DAGGER = 10     # 刀弹
    M_JADE = 11     # 中玉    (Middle Jade)
    B_STAR = 12     # 大星弹  (Big Star)
    ELLIPSE = 13
    # Giant
    G_JADE = 14     # 大玉    (Giant Jade)


class BulletColors(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3
    PUEPLE = 4
    WHITE = 5
#
#
# def is_bullet16(btype):
#     return btype.value < BulletTypes.SLIGHTJADE.value
#

def get_size(btype):
    if btype.value < 1:
        return Sizes.TINY
    elif btype.value < 10:
        return Sizes.MID
    elif btype.value < 14:
        return Sizes.BIG
    else:
        return Sizes.GIANT

def get_out_range(btype):
    size = get_size(btype)
    if size == Sizes.TINY:
        return [(-200, 200), (-232, 232)]
    elif size == Sizes.MID:
        return [(-208, 208), (-240, 240)]
    elif size == Sizes.BIG:
        return [(-224, 224), (-256, 256)]
    else:
        return [(-224, 224), (-256, 256)]

def get_render_layer(btype):
    size = get_size(btype)
    if size == Sizes.TINY:
        return FullRenderer.Layers.BulletTiny
    elif size == Sizes.MID:
        return FullRenderer.Layers.BulletMid
    elif size == Sizes.BIG:
        return FullRenderer.Layers.BulletBig
    else:
        return FullRenderer.Layers.BulletGiant

def get_weight(btype):
    size = get_size(btype)
    if size == Sizes.TINY:
        return 1.
    elif size == Sizes.MID:
        return 4.
    elif size == Sizes.BIG:
        return 16.
    else:
        return 64.

