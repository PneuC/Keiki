from utils.math import Vec2
from numpy import sin, cos, tan, radians


def sectoral_angles(dire=0.0, span=0, ways=1, radius=0.0):
    if radius == 0.0:
        if ways == 1:
            return [dire]
        elif ways > 1:
            lower_angle = dire - (span % 360) / 2
            angle_diff = span / (ways - 1)
            return [lower_angle + angle_diff * i for i in range(ways)]
    else:
        if ways == 1:
            return {'pos': Vec2.from_plr(radius, dire), 'angle': dire}
        elif ways > 1:
            lower_angle = dire - (span % 360) / 2
            angle_diff = span / (ways - 1)
            res = []
            for i in range(ways):
                item = {'angle': lower_angle + angle_diff * i}
                item['pos'] = Vec2.from_plr(radius, item['angle'])
                res.append(item)
            return res

def circular_configs(pos=None, angle=0.0, bend=0.0, radius=0.0, ways=1, es=False):
    if ways > 0:
        angle_diff = 360 / ways
        if es:
            angle += angle_diff / 2
        if radius == 0.0:
            return [angle + angle_diff * i + bend for i in range(ways)]
        else :
            if pos is None:
                pos = Vec2(0, 0)
            res = [
                {'pos': pos + Vec2.from_plr(radius, angle + angle_diff * i),
                 'angle': angle + angle_diff * i + bend
                } for i in range(ways)
            ]
            return res

def univals(v0, v1, n=1):
    ds = (v1 - v0) / (n - 1)
    return [v0 + ds * i for i in range(n)]

def chord(length, interior_angle, norm_angle=0, pos=None, ways=1):
    if pos is None:
        pos = Vec2(0, 0)
    if ways == 1:
        return pos
    norm = Vec2.from_plr(1, norm_angle)
    interior_angle %= 180
    if interior_angle <= 0.1:
        d = length / (ways - 1)
        v = Vec2(-norm.y, norm.x).norm()
        endpoint = pos - v * (length / 2)
        res = [
            {'pos': endpoint + v * (i * d), 'angle': norm_angle}
            for i in range(ways)
        ]
        return res
    else:
        if interior_angle > 175:
            raise RuntimeWarning('Too large interior angle of chord: %d' % interior_angle)
        l = 0.5 * length
        alpha = interior_angle / 2
        R = l / sin(radians(alpha))
        # print(alpha, tan(alpha), l / tan(alpha), norm_angle)
        O = pos - Vec2.from_plr(l / tan(radians(alpha)), norm_angle)
        lower_angle = norm_angle - alpha
        ad = interior_angle / (ways - 1)

        res = []
        for i in range(ways):
            a = lower_angle + i * ad
            config = {'pos': O + Vec2.from_plr(R, a), 'angle': a}
            res.append(config)
        return res

