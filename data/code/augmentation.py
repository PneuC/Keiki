from copy import copy
from utils.math import bounded_gauss


def transform_single(val):
    return val + bounded_gauss(sigma=0.05*val)

def transform_multi(*vals):
    return tuple(transform_single(val) for val in vals)

def transform_gl(vals, g_bias=0):
    transformed = copy(vals)
    for i in range(len(transformed)):
        transformed[i] += g_bias
        transformed[i] = transform_single(transformed[i])
    return tuple(transformed)

def transform_dicts(*argdicts):
    transformed = [argdict.copy() for argdict in argdicts]
    for item in transformed:
        for key in item.keys():
            orival = item[key]
            if type(orival) in {int, float}:
                item[key] = transform_single(orival)
    return tuple(transformed)

