import numpy as np
import numba
from PIL import Image
from math import log, floor, ceil
from scipy.interpolate import interp1d
import random

N_COLORS = 2000


@numba.jit
def mandel(c, max_iter):
    z = c
    for n in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n + 1 - log(log(abs(z)))/log(2)
        z = z*z + c
    return max_iter


@ numba.jit
def mandel_set(re_min, re_max, im_min, im_max, width, height, max_iter):
    c_map = np.zeros((height, width), dtype=np.cdouble)
    i_map = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            re = (x / (width-1)) * (re_max - re_min) + re_min
            im = (1 - y / (height-1)) * (im_max - im_min) + im_min
            c_map[y][x] = complex(re, im)

    for i, c in enumerate(c_map.flat):
        i_map.flat[i] = mandel(c, max_iter)

    return i_map


def colormap(controlpoints):
    i_c = [0.0, .6, .85, 1.0]

    xi = np.linspace(0, 1, N_COLORS)
    fc = interp1d(i_c, controlpoints)

    col_map = np.zeros((N_COLORS, 3))
    for i, x in enumerate(xi):
        col_map[i, :] = fc(x)

    return col_map


@ numba.jit
def colorize(i_map, max_iter, c_map):
    x, y = i_map.shape
    img = np.zeros((x, y, 3), dtype=np.uint8)

    hist = np.zeros(max_iter)
    for it in i_map.flat:
        if it < max_iter:
            hist[floor(it)] += 1
    total = hist.sum()

    c_range = np.zeros((max_iter + 1))
    h = 0
    for i in range(max_iter):
        h += hist[i] / total
        c_range[i] = h
    c_range[-1] = 1.0

    for i, row in enumerate(i_map):
        for j, it in enumerate(row):
            if it >= max_iter:
                img[i, j, :] = [0, 0, 0]
            else:
                idx = int(N_COLORS * lin_inter(
                    c_range[floor(it)], c_range[ceil(it)], it % 1))
                img[i, j, :] = c_map[idx, :]

    return img


@ numba.jit
def lin_inter(x1, x2, t):
    return x1 * (1 - t) + x2 * t


class Mandelbrot():

    def __init__(self, width=1000, height=800, Re_center=-0.75, Im_center=0, Re_width=2.5, max_iter=1000):
        self.width = width
        self.height = height
        self.re_width = Re_width
        self.re_min = Re_center - Re_width / 2
        self.re_max = Re_center + Re_width / 2
        self.im_height = height * Re_width / width
        self.im_min = Im_center - self.im_height / 2
        self.im_max = Im_center + self.im_height / 2
        self.max_iter = max_iter

    def compute_set(self):
        self.iterations = mandel_set(
            self.re_min, self.re_max, self.im_min, self.im_max,
            self.width, self.height, self.max_iter)

        return self.iterations

    def draw(self):
        self.im = Image.fromarray(
            colorize(self.iterations, self.max_iter, self.cmap), 'RGB')

        return self.im

    def create_colormap(self, seed=-1):
        controlpoints = np.zeros((3, 4))
        if seed == -1:
            # default colormap
            self.seed = -1

            controlpoints[:, :] = [[67, 250, 0, 67],
                                   [35, 174, 0, 35],
                                   [113, 123, 0, 113]]
        else:
            if seed == 0:
                random.seed()
                self.seed = random.randint(1, 1000000)
                random.seed(self.seed)
                # Random seed
            else:
                # Specified seed
                self.seed = seed
                random.seed(seed)

            controlpoints[:, 0] = [random.randint(0, 255) for i in range(3)]
            controlpoints[:, -1] = controlpoints[:, 0]

            controlpoints[:, 1] = [random.randint(0, 255) for i in range(3)]

        self.cmap = colormap(controlpoints)
        return self.seed

    def save(self, path):
        self.im.save(path)
