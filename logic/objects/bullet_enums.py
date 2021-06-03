from enum import Enum
from render.real_time.renderer import FullRenderer


class Sizes(Enum):
    TINY = 8
    MID = 16
    BIG = 32
    GIANT = 64


class BulletTypes(Enum):
    # Tiny
    DOT = 0
    # Mid
    DROP = 1
    STAR = 2
    CORN = 3
    BULLET = 4
    OFUDA = 5
    KUNAI = 6
    S_JADE = 7
    R_JADE = 8
    SQUAMA = 9
    # Big
    DAGGER = 10
    M_JADE = 11     # (Middle Jade)
    B_STAR = 12     # (Big Star)
    ELLIPSE = 13
    # Giant
    G_JADE = 14     # (Giant Jade)


class BulletColors(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3
    PUEPLE = 4
    WHITE = 5


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

