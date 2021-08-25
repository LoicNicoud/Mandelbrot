from time import time
import click
from .mandelbrot import Mandelbrot


@click.command(help='Compute and colorize the mandelbrot set around a certain point')
@click.option('-w', '--width',
              default=6000, type=int, show_default=True,
              help='Width of the output image')
@click.option('-h', '--height',
              default=3200, type=int, show_default=True,
              help='Height of the output image')
@click.option('-rc', '--real-center',
              default=-0.743643887037151, type=float, show_default=True,
              help='Real part of the center point')
@click.option('-ic', '--imag-center',
              default=0.13182590420533, type=float, show_default=True,
              help='Imaginary part of the center point')
@click.option('-rw', '--real-width',
              default=0.0015, type=float, show_default=True,
              help='wiew width of the set')
@click.option('-m', '--max-iter',
              default=500, type=int, show_default=True,
              help='Maximum iterations per point')
@click.option('-s', '--seed',
              default=0, type=int, show_default=True,
              help='Seed for the colorgradient, -1 means the default colormap, 0 a random seed, or the seed number')
@click.option('-o', '--out',
              default='images/mandel.png', type=click.Path(), show_default=True,
              help='output image path')
def main(width, height, real_center, imag_center, real_width, max_iter, out, seed):
    mandel = Mandelbrot(width, height, real_center,
                        imag_center, real_width, max_iter)

    start_time = time()
    click.echo('Computing set')
    mandel.compute_set(seed=seed)
    click.echo(f'Done in :{time() - start_time:.4f}s.')

    # start_time = time()
    # click.echo('Creating color map')
    # seed = mandel.create_colormap(seed)
    # click.echo(f'Done in :{time() - start_time:.4f}s.')

    # start_time = time()
    # click.echo(f'Colorizing set with seed: {seed}')
    # mandel.draw()
    # click.echo(f'Done in :{time() - start_time:.4f}s.')

    mandel.save(out)
    click.echo(f'Image saved to: {out}')


main()
