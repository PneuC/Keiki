from numpy import radians, degrees, cos, sin, arctan
from random import gauss


def bound(x, lb, ub):
    return min(max(x, lb), ub)

def bounded_gauss(mu=0, sigma=1, factor=3):
    lb, ub = mu - factor * sigma, mu + factor * sigma
    return bound(gauss(mu, sigma), lb, ub)

def vcos(u, v):
    a = u * v
    b = u.magnitude * v.magnitude
    return a / b

def sin_cos_convert(val):
    return (1 - val *val) ** 0.5

def linear_mapping(domain, vrange, x):
    # if not domain[0] <= x <= domain[1]:
    #     raise ValueError('Linear Mapping: x value (%.10f) exceed domain (%.1f, %.1f)' % (x, domain[0], domain[1]))
    x = bound(x, domain[0], domain[1])
    scale = vrange[1] - vrange[0]
    rate = (x - domain[0]) / (domain[1] - domain[0])
    return vrange[0] + rate * scale

class Vec2:
    def __init__(self, *args):
        self.x = args[0]
        self.y = args[1]

    @staticmethod
    def from_plr(r, w):
        return Vec2.from_std_plr(r, w - 90)

    @staticmethod
    def from_std_plr(r, w):
        w = radians(w)
        return Vec2(r * cos(w), r * sin(w))

    def magnitude(self):
        """
        Get the magnitude of current instance.

        :return: magnitude in float
        """
        m = self.x * self.x + self.y * self.y
        return m ** 0.5

    def do_norm(self):
        """
        Do (L2) normalization on current instance.

        :return: None
        """
        m = self.magnitude()
        if m != 0:
            self.x /= m
            self.y /= m

    def norm(self):
        """
        Get the (L2) normalized vector of current instance.

        :return: normalized vector in Vec2
        """
        m = self.magnitude()
        if m == 0:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / m, self.y / m)

    def is_out(self, x_range, y_range):
        """
        Check current instance is out of a range or not.

        :param x_range: range of x in the form [min, max]
        :param y_range: range of y in the form [min, max]
        :return: out or not
        """
        x_out = not x_range[0] <= self.x <= x_range[1]
        y_out = not y_range[0] <= self.y <= y_range[1]
        return x_out or y_out

    def do_bound_in(self, x_range, y_range):
        """
        Bound the current instance in a range.
        In the other word, do max and min clip on both x and y.

        :param x_range: range of x in the form [min, max]
        :param y_range: range of y in the form [min, max]
        :return: None
        """
        self.x = max(self.x, x_range[0])
        self.x = min(self.x, x_range[1])
        self.y = max(self.y, y_range[0])
        self.y = min(self.y, y_range[1])

    def bound_in(self, x_range, y_range):
        """
        Get the vector which after doing max and min clip on both x and of current instance.

        :param x_range: range of x in the form [min, max]
        :param y_range: range of y in the form [min, max]
        :return: a new vector
        """
        x = max(self.x, x_range[0])
        x = min(x, x_range[1])
        y = max(self.y, y_range[0])
        y = min(y, y_range[1])
        return Vec2(x, y)

    def to_rcs(self):
        """
        To render coordinate system.

        :return: The corresponding render coordinate.
        """
        return Vec2(192 + self.x, 224 - self.y)

    def std_theta(self):
        if self.x:
            d = degrees(arctan(self.y / self.x))
        else:
            d = -90.0
        return d % 360 if self.x >= 0 else d + 180

    def theta(self):
        return (self.std_theta() + 90) % 360

    def set_mag(self, mag):
        ratio = mag / self.magnitude()
        self.x *= ratio
        self.y *= ratio

    def cpy(self):
        return Vec2(self.x, self.y)

    def tuple(self):
        return self.x, self.y

    def __getattr__(self, item):
        if item in {'r', 'm', 'magnitude'}:
            return self.magnitude()

    def __setattr__(self, key, value):
        if key in {'r', 'm', 'magnitude'}:
            self.set_mag(float(value))
        else:
            super().__setattr__(key, value)

    def __add__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)
        else:
            result = self.cpy()
            result.set_mag(self.m + other)
            return result

    def __sub__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x - other.x, self.y - other.y)
        else:
            result = self.cpy()
            result.set_mag(self.m - other)
            return result

    def __mul__(self, other):
        if type(other) == Vec2:
            return self.x * other.x + self.y * other.y
        else:
            return Vec2(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return Vec2(self.x / other, self.y / other)

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def __abs__(self):
        return Vec2(abs(self.x), abs(self.y))

    def __str__(self):
        return '(%.3f, %.3f)' % (self.x, self.y)

    def tran_base_Xaxis(self, Xaxis):
        if isinstance(Xaxis, Vec2):
            x = self * Xaxis / Xaxis.magnitude()
            vsqr = self.x * self.x + self.y * self.y
            diff = vsqr - x * x
            if diff > 0.:
                y = diff ** 0.5
            else:
                y = 0.
            return Vec2(x, y)
        else:
            return self.tran_base_Xaxis(Vec2.from_plr(1, Xaxis))

if __name__ == '__main__':
    print(linear_mapping((0, 10), (0, 1), 3))
    print(linear_mapping((0, 10), (0, 1), 0))
    print(linear_mapping((0, 10), (0, 1), 10))
    print(linear_mapping((0, 10), (0, 1), 7))
    # print(bound(0, -1, 1))
    # print(bound(3, -1, 1))
    # print(bound(-3, -1, 1))

