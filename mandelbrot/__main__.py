from .mandelbrot import Mandelbrot
import click


@click.command(help='Compute and colorize the mandelbrot set around a certain point')
@click.option('-w', '--width',
              default=1500, type=int, show_default=True,
              help='Width of the output image')
@click.option('-h', '--height',
              default=1000, type=int, show_default=True,
              help='Height of the output image')
@click.option('-rc', '--real-center',
              default=-0.7106, type=float, show_default=True,
              help='Real part of the center point')
@click.option('-ic', '--imag-center',
              default=+0.246575, type=float, show_default=True,
              help='Imaginary part of the center point')
@click.option('-rw', '--real-width',
              default=0.0084, type=float, show_default=True,
              help='wiew width of the set')
@click.option('-m', '--max-iter',
              default=1500, type=int, show_default=True,
              help='Maximum iterations per point')
@click.option('-s', '--seed',
              default=0, type=int, show_default=True,
              help='Seed for the colorgradient, -1 means the default colormap, 0 a random seed, or the seed number')
@click.option('-o', '--out',
              default='images/mandel.jpg', type=click.Path(), show_default=True,
              help='output image path')
def main(width, height, real_center, imag_center, real_width, max_iter, out, seed):
    mandel = Mandelbrot(width, height, real_center,
                        imag_center, real_width, max_iter)

    click.echo('Computing set')
    mandel.compute_set()

    seed = mandel.create_colormap(seed)
    click.echo(f'Colorizing set with seed: {seed}')
    mandel.draw()

    mandel.save(out)
    click.echo(f'Image saved to: {out}')


main()
