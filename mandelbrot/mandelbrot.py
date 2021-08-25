import numpy as np
import numba
from PIL import Image
from math import log, floor, ceil, sin
from scipy.interpolate import interp1d
import random
import pyopencl as cl
import pyopencl.array as cl_array
import time

N_COLORS = 2000


def comp_mandelset_opencl(cgrid: np.ndarray, maxiter: int, ampl, ofst, per):
    """Compute the mandelbrot set iteration with pyopencl."""
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # out = np.empty(cgrid.shape, dtype=np.double)
    out = np.empty(cgrid.shape, dtype=np.uint32)

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cgrid)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)

    # ampl_buff = cl.Buffer(ctx, mf.READ_ONLY, ampl.nbytes)
    # ofst_buff = cl.Buffer(ctx, mf.READ_ONLY, ofst.nbytes)
    # per_buff = cl.Buffer(ctx, mf.READ_ONLY, per.nbytes)
    ampl_dev = cl_array.to_device(queue, ampl)
    ofst_dev = cl_array.to_device(queue, ofst)
    per_dev = cl_array.to_device(queue, per)

    prg = cl.Program(
        ctx,
        """

    __kernel void mandelbrot(__global double2 *q,
                     __global uchar4 *output, ushort const maxiter, 
                     __global double *ampl, __global double *ofst, __global double *per)
    {
        int gid = get_global_id(0);
        double x = 0.0f;
        double y = 0.0f;
        double x2 = 0.0f;
        double y2 = 0.0f;
        double w = 0.0f;
        uint curiter = 0;
        double normiter = 0.0f;

        output[gid] = 0.0f;

        while ((x2 + y2 <= 4.0f) && (curiter < maxiter)) {
            y = (x + x) * y + q[gid].y;
            x = x2 - y2 + q[gid].x;
            x2 = x * x;
            y2 = y * y;
            curiter++;
        }

        if (curiter < maxiter) {
            normiter = (float) curiter;
            normiter += 1 - log(log2(sqrt(x2 + y2)));
            output[gid].x = (int) (sin(per[0] * normiter) * ampl[0] + ofst[0]);
            output[gid].y = (int) (sin(per[1] * normiter) * ampl[1] + ofst[1]);
            output[gid].z = (int) (sin(per[2] * normiter) * ampl[2] + ofst[2]);
            output[gid].w = 255;
        }
        else {
            output[gid].x = 0;
            output[gid].y = 0;
            output[gid].z = 0;
            output[gid].w = 0;
        }

        

    }
    """,
    ).build()

    prg.mandelbrot(
        queue, out.shape, None, q_opencl, output_opencl, np.uint16(
            maxiter), ampl_dev.data, ofst_dev.data, per_dev.data
    )

    cl.enqueue_copy(queue, out, output_opencl).wait()
    # byt = out.tobytes()
    # out = np.frombuffer(byt, dtype=np.uint8)

    return out


def colorize(iter_map, max_iter, v_freq, v_phase):
    hi, wi = iter_map.shape
    img = np.zeros((hi, wi, 3), dtype=np.uint8)

    for x in range(wi):
        for y in range(hi):
            it = iter_map[y, x]
            if it >= max_iter:
                img[y, x] = [0, 0, 0]
            else:
                img[y, x] = sin_col(it, v_freq, v_phase)

    return img


def lin_inter(x1, x2, t):
    return x1 * (1 - t) + x2 * t


def sin_col(iter, v_freq, v_phase):
    return [sin(f * iter + p) * 127.5 + 127.5 for f, p in zip(v_freq, v_phase)]


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

    def compute_set(self, seed):
        self._init_colorval(seed)

        xx = np.linspace(self.re_min, self.re_max, num=self.width)
        yy = np.linspace(self.im_min, self.im_max, num=self.height) * 1j
        q = np.ravel(xx + yy[:, np.newaxis]).astype(np.complex128)
        print(q.shape)

        start_main = time.time()
        output = comp_mandelset_opencl(
            q, self.max_iter, self.amp, self.ofst, self.per)
        end_main = time.time()

        secs = end_main - start_main
        print("Main took", secs)

        self.im = Image.fromarray(output.reshape(
            (self.height, self.width)), 'RGBA')

        return self.im

    def draw(self):
        self.im = Image.fromarray(
            colorize(self.iterations, self.max_iter, [0.008, 0.006, 0.004], [4, 2, 1]), 'RGB')

        return self.im

    def _init_colorval(self, seed=0):
        if seed == 0:
            random.seed()
            self._seed = random.randint(1, 1000000)
            random.seed(self._seed)

        else:
            self._seed = seed
            random.seed(seed)

        # Gen 3 period
        self.per = np.asarray([random.uniform(0.01, 0.1)
                              for i in range(3)], dtype=np.float64)

        # Gen amplitude
        self.amp = np.asarray([random.uniform(0.0, 127.5)
                              for i in range(3)], dtype=np.float64)

        # gen offset
        self.ofst = np.asarray([random.uniform(i, 255.0-i)
                               for i in self.amp], dtype=np.float64)

    def create_colormap(self, seed=-1):
        controlpoints = np.zeros((3, 4))
        controlpos = [0.0, .6, .85, 1.0]
        if seed == -1:
            # default colormap
            self.seed = -1

            # controlpoints[:, :] = [[67, 250, 0, 67],
            #                        [35, 174, 0, 35],
            #                        [113, 123, 0, 113]]

            controlpoints = [[0, 0, 10, 148, 233, 238, 202, 187, 174, 155],
                             [18, 95, 147, 210, 216, 155, 103, 62, 32, 34],
                             [25, 115, 150, 189, 166, 0, 2, 3, 18, 38]]
            controlpos = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]

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

            colornum = random.randint(3, 10)
            controlpoints = np.zeros((colornum, 3))
            controlpos = np.linspace(0.0, 1.0, colornum)

            for color in controlpoints:
                color[:] = [random.randint(0, 255) for i in range(3)]

            controlpoints = np.transpose(controlpoints)

        self.cmap = colormap(controlpoints, controlpos)
        return self.seed

    def save(self, path):
        self.im.save(path)
