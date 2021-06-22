# Mandelbrot

Create an image of the mandelbrot set

## Installing dependencies

```console
$ pip3 install -r requirements.txt
```

## Examples

---

```console
$ python3 -m mandelbrot -w 3000 -h 2000 -s -1 -o images/Example1.jpg
```

<img src="images/Example1.jpg" alt="Example2" width="800"/>

---

```console
$ python3 -m mandelbrot -w 3000 -h 2000 -rc -0.170337 -ic -1.06506 -rw 0.1 -s -1 -o images/Example2.jpg
```

<img src="images/Example2.jpg" alt="Example2" width="800"/>