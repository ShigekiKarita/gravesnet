from chainer import functions
import numpy
from src.sqrt import mysqrt
from src.vec_sub_mat import vec_sub_mat


def gaussian_mixture_2d_ref(*inputs):
    w, m1, m2, s1, s2, c, x1, x2 = inputs
    z1 = vec_sub_mat(x1, m1, lhs_bwd=False) / s1
    z2 = vec_sub_mat(x2, m2, lhs_bwd=False) / s2
    z1 = (z1 - c * z2)**2.0
    z2 = 1.0 - c**2.0
    z3 = 2.0 * numpy.pi * s1 * s2 * z2 ** 0.5
    r = w * functions.exp(- z1 / (2.0 * z2)) / z3
    return r