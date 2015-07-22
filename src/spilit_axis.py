import numpy
from chainer import cuda
from chainer import functions as F

def to_indices(widths):
    """
    >>> to_indices([1, 2, 3, 1])
    [1, 3, 6]
    """
    i = 0
    indices = []
    for w in widths[:-1]:
        i += w
        indices.append(i)
    return indices


def split_axis_by_widths(x, widths, axis=1):
    if isinstance(widths, int):
        n = int(x.data.shape[axis] / widths)
        widths = [n] * widths
    indices = to_indices(widths)
    return F.split_axis(x, indices, axis)