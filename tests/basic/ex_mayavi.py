import numpy as np
from mayavi.mlab import points3d
from mayavi import mlab
import time 
import math

@mlab.show
def test_points3d():
    t = np.linspace(0, 4 * np.pi, 20)

    x = np.sin(2 * t)
    y = np.cos(t)
    z = np.cos(2 * t)
    s = 2 + np.sin(t)

    return points3d(x, y, z, s, colormap="copper", scale_factor=.25)

# to show the above example uncomment
# test_points3d()


# a simple animation example
# x, y = np.mgrid[0:3:1,0:3:1]
# s = mlab.surf(x, y, np.asarray(x*0.1, 'd'))

# @mlab.animate
# def anim():
#     for i in range(10):
#         s.mlab_source.scalars = np.asarray(x*0.1*(i+1), 'd')
#         yield

# anim()
# mlab.show()

##############3 cool example below! #################3

n_mer, n_long = 6, 11
pi = np.pi
dphi = pi/1000.0
phi = np.arange(0.0, 2*pi + 0.5*dphi, dphi, 'd')
mu = phi*n_mer
x = np.cos(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
y = np.sin(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
z = np.sin(n_long*mu/n_mer)*0.5

# View it.
l = mlab.plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap='Spectral')

# Now animate the data.
@mlab.animate(delay=10)
def anim():
    ms = l.mlab_source
    for i in np.linspace(0,60,1000):
        x = np.cos(mu)*(1+np.cos(n_long*mu/n_mer +
                                        np.pi*(i+1)/5.)*0.5)
        scalars = np.sin(mu + np.pi*(i+1)/5)
        # print(x, scalars)
        ms.trait_set(x=x, scalars=scalars)
        yield
anim()
mlab.show()

################## Another example #####################3

def produce_verts(A, t):
    def delta(A, t):
        return A * math.sin(t)
    def verts(d):
        return [(1 + d, 0, 0), (0, 1 + d, 0), (-1 - d, 0, 0), (0, -1 - d, 0),
                (0, 0, 1 + d), (0, 0, -1 - d)]
    return zip(*verts(delta(A, t)))

@mlab.animate(delay=100)
def anim():
    t = 0.
    dt = 0.1
    A = 0.5
    f = mlab.gcf()
    nverts = 6
    x, y, z = produce_verts(A, t)
    # Each triangle is a 3-tuple of indices. The indices are indices of `verts`.
    triangles = [(i, (i + 1) % 4, j) for i in range(4) for j in (4, 5)]
    colorval = [xi ** 2 + yi ** 2 + zi ** 2 for xi, yi, zi in zip(x, y, z)]
    mesh = mlab.triangular_mesh(
        x, y, z, triangles, scalars=colorval, opacity=1, representation='mesh')
    ms = mesh.mlab_source
    while True:
        f.scene.camera.azimuth(10)
        f.scene.render()
        t = (t + dt) % (2 * math.pi)
        x, y, z = produce_verts(A, t)
        colorval = [xi ** 2 + yi ** 2 + zi ** 2 for xi, yi, zi in zip(x, y, z)]
        ms.set(x=x, y=y, z=z, scalars=colorval)
        # time.sleep(0.1)
        print(t, dt)
        if t > 4:
            break
        yield
anim()
mlab.show()
