"""The following module contains numba-jitted 2d and 3d grid graphs,
which make using neighbors and cost function fast. The major benefit
is being able to use numba-jitted search (priority queue).

"""
import numba as nb
from numba import typed, typeof, objmode
from numba.experimental import jitclass
import numpy as np
import logging
from typing import List

import matplotlib.pyplot as plt


# l = logging.getLogger("numba")
# l.setLevel(logging.debug)

import os

# Set environment variables
# os.environ['NUMBA_DEBUG'] = "1"

# constants, the first is based on grid_size 
vl = 1
SQ3 = 1.7320508075688772
SQ2 = 1.4142135623730951
C1 = SQ3 - SQ2
C2 = SQ2 - vl
C3 = vl

# specify for grid dim
list_instance = typed.List([0]*6)

spec = [("domain_type", nb.types.unicode_type),
        ("edge_type", nb.types.unicode_type),
        ('grid_size', nb.types.int64),
        # ('grid_size', nb.types.float64),
        ('grid_dim', typeof(list_instance)),
        ('x_len', nb.types.int16),
        ('y_len', nb.types.int16),
        ('z_len', nb.types.int16),
        ('coord_dims', nb.types.UniTuple(nb.types.int16, 3)),
        ('grid', nb.types.float64[:,:,:]),
]

samples_def = nb.typeof(nb.typed.List.empty_list((0,0,0)) )

@jitclass(spec)
class RectGrid3D:
    """a 3d based grid class using numpy array as an underlying data structure
    
    Assume a unit grid size for now and origin of (0,0,0)

    Add support to differentiate indices and world coordinates
    
    """


    def __init__(self, grid_dim: list, obstacles: list=None):
        # self.name = None
        self.domain_type = "GRID_3D"
        self.edge_type = "UNDIRECTED"

        # assume unit grid spacing    
        self.grid_size = 1
        self.grid_dim = grid_dim
        self.x_len = int((grid_dim[1] - grid_dim[0] + 1))
        self.y_len = int((grid_dim[3] - grid_dim[2] + 1))
        self.z_len = int((grid_dim[5] - grid_dim[4] + 1))

        # create a 3d np array
        self.coord_dims = (self.x_len, self.y_len, self.z_len)
        self.grid = np.zeros(self.coord_dims, dtype=np.float64)

        # set cells to 1 if obstacle. no need to store obstacles list
        # self.obstacles = obstacles 
        if obstacles is not None:
            for o in obstacles:
                x,y,z = o
                self.grid[x][y][z] = 1
        # self.obs = np.array(obstacles)
        # self.grid[self.obs[:,0], self.obs[:,1], self.obs[:,2]] = 1

        # C2 = SquareGrid3D.SQ2 - grid_size
        # C3 = grid_size

    def in_bounds(self, node):
        """node must fall within bounds"""
        x,y,z = node
        return self.grid_dim[0] <= x <= self.grid_dim[1] and \
               self.grid_dim[2] <= y <= self.grid_dim[3] and \
               self.grid_dim[4] <= z <= self.grid_dim[5]

    def not_obstacles(self, start, node):
        """node coordinates coincide with cell index
        
        during diagonal actions, don't allow cutting of corners, i.e. each individual
        cardinal action must also be possible.

        """
        x = int(node[0])
        y = int(node[1])
        z = int(node[2])

        # xs, ys, zs = start
        xs = int(start[0])
        ys = int(start[1])
        zs = int(start[2])

        # get vector to destination node
        dx = x - xs
        dy = y - ys
        dz = z - zs
        # check cardinal directions
        res = nb.types.boolean
        res = self.grid[xs + dx,ys,zs] < 1 and self.grid[xs,ys + dy,zs] < 1 and self.grid[xs,ys,zs + dz] < 1

        # check target location
        return self.grid[x,y,z] == 0 and res

    # def neighbors(self, node:tuple):
    #     """wrap tuple"""

    def neighbors(self, node):
        """There are at most 26 neighbors, subject to obstacles"""
        x = int(node[0])
        y = int(node[1])
        z = int(node[2])

        if self.grid[x,y,z] > 0:
            return None

        layer_wo_mid = lambda z: [(x + self.grid_size, y, z), (x, y - self.grid_size, z),
                       (x - self.grid_size, y, z), (x, y + self.grid_size, z),
                       (x + self.grid_size, y + self.grid_size, z), (x + self.grid_size, y - self.grid_size, z),
                       (x - self.grid_size, y - self.grid_size, z), (x - self.grid_size, y + self.grid_size, z)]

        results = []
        # Add each layer without a middle node
        results.extend(layer_wo_mid(z-self.grid_size))
        results.extend(layer_wo_mid(z))
        results.extend(layer_wo_mid(z+self.grid_size))
        # add two missing middle nodes
        results.extend([(x, y, z - self.grid_size), (x ,y, z + self.grid_size)]) 

        results2 = []
        # apply filter to remove nodes
        for v in results:
            if self.in_bounds(v):
                results2.append(v)

        results = [] 
        for v in results2:
            if self.not_obstacles(node, v):
                results.append(v)

        # results = filter(lambda x: self.in_bounds(x), results) 
        # results = filter(lambda x: self.not_obstacles(node, x), results) 

        return results

    def cost(self, from_node, to_node):
        """ based on voxel distance (octile distance generalized to 3d)"""
        x1, y1, z1 = from_node
        x2, y2, z2 = to_node
        dx, dy, dz = abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)
        dmax = max(dx, dy, dz)
        dmin = min(dx, dy, dz)
        dmid = dx + dy + dz - dmin - dmax 
        return C1*dmin + C2*dmid + C3*dmax
    
    # def show_grid(self):
    #     ax = plt.figure().add_subplot(projection='3d')
    #     ax.voxels(self.grid, facecolors="red", edgecolor='k')
    #     plt.show(block=False)

    def node_count(self):
       return np.count_nonzero(self.grid==0) 
    
    def edge_count(self):
        """LuLs not implemented, not needed"""
        pass

    def sample_uniform(self, num_of_samples: int):
        """uniformly sample the free space"""
        # min_x, max_x, min_y, max_y, min_z, max_z = self.grid_dim
        # samples = set()
        # while len(samples) < min(num_of_samples, self.node_count()):
        #     # blah = np.random.randint((min_x, min_y, min_z), (max_x, max_y, max_z))
        #     x = np.random.randint(min_x, max_x)
        #     y = np.random.randint(min_y, max_y)
        #     z = np.random.randint(min_z, max_z)
        #     if self.grid[x,y,z] == 0:
        #         # add samples as tuples
        #         samples.add((x,y,z))
        
        # samples = nb.typed.List.empty_list((0,0,0)) 
        # with objmode(samples=samples_def):

        samples = np.empty((num_of_samples, 3),dtype=nb.int64)
        with objmode(samples='int64[:,:]'):
            min_x, max_x, min_y, max_y, min_z, max_z = self.grid_dim
            samples = set()
            current_size = num_of_samples
            while len(samples) < num_of_samples:
                # randomly generate N number of indices samples
                # gen = np.empty((current_size, 3))
                # gen = np.random.randint((min_x, min_y, min_z), (max_x, max_y, max_z), size=(current_size,3))
                # return random integers from low (inclusive) to high (exclusive)
                gen = np.random.randint(low=(min_x, min_y, min_z), high=(max_x+1, max_y+1, max_z+1), size=(current_size,3))
                # non-obstacle cell mask
                get = self.grid[gen[:,0], gen[:,1], gen[:,2]] < 1
                # grab all non-obstacle indices
                items = gen[get]
                # add to samples set for uniquenes
                for i in items:
                    if len(samples)<num_of_samples:
                        samples.add(tuple(i))
                    else:
                        break
                current_size = num_of_samples - len(samples)
            samples = list(samples)
            samples = np.array(samples)

            # conver to nb typed list
            # samples = nb.typed.List.empty_list((0,0,0))
            # for s in _samples:
            #     samples.append(s)

            # return list(samples)

        # conver to a list of 3-tuples before returning
        _s = list()
        for s in samples:
            _s.append((s[0], s[1], s[2]))
        return _s
        # return samples

# 2d samples
samples_def = nb.typeof(nb.typed.List.empty_list((0,0)) )
grid_dim_def = nb.typeof(nb.typed.List.empty_list(nb.types.int64))

@jitclass
class RectGrid2D:
    """Assume 8 neighbor type grid graph, with at unit edge cost
        between cardinal directions

    """
    domain_type: nb.types.unicode_type
    edge_type: nb.types.unicode_type
    grid_dim: grid_dim_def
    grid_size: nb.types.int64
    xwidth: nb.types.int64
    # obstacles: nb.typeof(nb.typed.List.empty_list(nb.types.UniTuple(nb.types.int64,2)))
    yheight: nb.types.int64
    coord_dim: nb.typeof(nb.types.UniTuple(nb.types.int16,2))
    grid: nb.types.float64[:,:]

    def __init__(self, grid_dim: list, obstacles: list=None):
        
        self.domain_type = "GRID_2D"
        self.edge_type = "UNDIRECTED"

        self.grid_dim = grid_dim
        self.grid_size = 1

        self.xwidth = int(grid_dim[1]-grid_dim[0] + 1)
        self.yheight = int(grid_dim[3]-grid_dim[2] + 1)

        coord_dim = (self.xwidth, self.yheight)

        # self.obstacles = obstacles

        # self.grid = np.zeros((self.xwidth, self.yheight))
        self.grid = np.zeros(coord_dim)
        for o in obstacles:
            self.grid[o] = 1

    def obstacles(self):
        """Return obstacles as a numpy array"""
        return np.where(self.grid==1)
        # x,y= np.where(self.grid==1)
        # return (y,x)

     
    def node_count(self):
        return np.count_nonzero(self.grid==0)
        # return len(self.grid<1)
        
    def in_bounds(self, node):
        (x,y) = node
        return self.grid_dim[0] <= x <= self.grid_dim[1] and self.grid_dim[2] <= y <= self.grid_dim[3]

    def not_obstacles(self, node):
        return self.grid[node] < 1

    def neighbors(self, node):
        x = int(node[0])
        y = int(node[1])

        results = [(x + self.grid_size, y), (x, y - self.grid_size),
                    (x - self.grid_size, y), (x, y + self.grid_size),
                    (x + self.grid_size, y + self.grid_size), (x + self.grid_size, y - self.grid_size),
                    (x - self.grid_size, y - self.grid_size), (x - self.grid_size, y + self.grid_size)]

        results2 = []
        for r in results:
            if self.in_bounds(r):
                results2.append(r)

        results = []
        for r in results2:
            if self.not_obstacles(r):
                results.append(r)
        
        return results

    def cost(self, from_node, to_node):
        """edge cost based on grid-based calculations """
        (x1, y1) = from_node
        (x2, y2) = to_node
        dmax = max(abs(x1 - x2), abs(y1 - y2))
        dmin = min(abs(x1 - x2), abs(y1 - y2))
        return 1.414*dmin + (dmax - dmin)

    def show_grid(self):
        """A debugging method to display the current occupancy grid.
        Don't keep the plot opened outside of debugging

        WARNING: may cause hang ups
        
        """
        with objmode():

            # Get grid dims
            minX, maxX, minY, maxY = self.grid_dim

            # get fig, ax objects
            fig = plt.figure()
            ax = fig.gca()

            # plotting in a non-blocking manner
            # plt.ion()
            # plt.draw()

            plt.pause(0.75)

            fig.canvas.draw_idle() 
            plt.show(block=False)

            # if interacting with canvas directly, must flush events after drawing
            # self.fig.canvas.flush_events()
            background = fig.canvas.copy_from_bbox(ax.bbox)

            # show grid as an image i.e. on a 2d raster
            # cmap _r indicates reversed (i.e. Blues_r, Black_r)
            # plt.draw()
            im = ax.imshow(
                self.grid.T,
                origin='lower',
                interpolation='none',
                alpha=1,
                vmin=0,
                vmax=1,
                extent=[
                    minX-self.grid_size/2,
                    maxX+self.grid_size/2,
                    minY-self.grid_size/2,
                    maxY+self.grid_size/2],
                cmap='Blues', aspect='equal')

            # xmin, xmax, ymin, ymax = self.grid_dim
            # self.ax.set_xticks(np.arange(xmin, xmax+1, self.grid_size))
            # self.ax.set_yticks(np.arange(ymin, ymax+1, self.grid_size))
            # self.ax.set_xticklabels(np.arange(xmin, xmax+1, self.grid_size))
            # self.ax.set_yticklabels(np.arange(ymin, ymax+1, self.grid_size))
            
            plt.title("Occupancy Grid Map")
            plt.axis('scaled')  #equal is another one
            # plt.grid()
            
            fig.canvas.restore_region(background)
            # Draw artists on helper objects
            ax.draw_artist(im)

            fig.canvas.blit(ax.bbox)
            # fig.canvas.update()

            # must call fig.canvas.flush_events() (called by pause internally)
            fig.canvas.flush_events()
            # plt.pause(0.5)
            # plt.show(block=True)


    def sample_uniform(self, num_of_samples: int):
        """uniformly sample the free space
        
        It is faster to go back to objmode to use randint's size parameter
        than to make repeated calls to randint

        """
        samples = np.empty((num_of_samples, 2),dtype=nb.int64)
        with objmode(samples='int64[:,:]'):
            min_x, max_x, min_y, max_y = self.grid_dim
            samples = set()
            # ensure the num of samples does not exceed node count
            num_of_samples = min(num_of_samples, self.node_count())
            # current_size is adaptive
            current_size = num_of_samples
            while len(samples) < num_of_samples:
                # randomly generate N number of indices samples
                # gen = np.empty((current_size, 3))
                # gen = np.random.randint((min_x, min_y, min_z), (max_x, max_y, max_z), size=(current_size,3))
                # return random integers from low (inclusive) to high (exclusive)
                gen = np.random.randint(low=(min_x, min_y), high=(max_x+1, max_y+1), size=(current_size,2))
                # non-obstacle cell mask
                get = self.grid[gen[:,0], gen[:,1]] < 1
                # grab all non-obstacle indices
                items = gen[get]
                # add to samples set for uniquenes
                # print(current_size)
                for i in items:
                    if len(samples)<num_of_samples:
                        samples.add(tuple(i))
                    else:
                        break
                # current_size = num_of_samples - len(samples)
            samples = list(samples)
            samples = np.array(samples)

            # conver to nb typed list
            # samples = nb.typed.List.empty_list((0,0,0))
            # for s in _samples:
            #     samples.append(s)

            # return list(samples)

        # conver to a list of 3-tuples before returning
        _s = list()
        for s in samples:
            _s.append((s[0], s[1])) 
        return _s

def grid2d_wrap(grid_dim: list, obstacles: List[tuple]):
    """wrapper around Rect2D jitted object"""
    # nb_dim_list = nb.typed.List([0]*4)
    nb_dim_list = nb.typed.List(grid_dim)
    if obstacles:
        nb_obs_list = nb.typed.List(obstacles)
    else:
        nb_obs_list = nb.typed.List.empty_list(nb.types.UniTuple(nb.types.int64,2))
        # obstacles: nb.typeof(nb.typed.List.empty_list(nb.types.UniTuple(nb.types.int64,2)))
    # # convert python list to numba list
    # for ndx, d in enumerate(grid_dim):
    #     nb_dim_list[ndx] = d

    return RectGrid2D(nb_dim_list, nb_obs_list)
    # return RectGrid2D(nb_dim_list, obstacles)


def grid3d_wrap(grid_dim, obstacles):
    """Wrapper around Rect3d jitted object"""

    # convert obstacles to typed list
    obs = nb.typed.List(obstacles)
    if obstacles:
        obs = nb.typed.List(obstacles)
    else:
        obs = nb.typed.List.empty_list(nb.types.UniTuple(nb.types.int64,3))

    for i, v in enumerate(grid_dim):
        list_instance[i] = v

    return RectGrid3D(list_instance, obs)

if __name__ == "__main__":
    "try to create numba stuff"
    for i, v in enumerate([0, 99, 0, 99, 0, 99]):
        list_instance[i] = v

    obs = nb.typed.List([(0,0,0)])
    ro = RectGrid3D(list_instance, obs) 
    pass
    # nb.typeof(nb.typed.List.empty(nb.types.int64, 4))

    # try 2d grid
    # dimensions
    dim = [0, 99, 0, 99]
    obs = [(0,0)]

    ro2 = grid2d_wrap(dim, obs)
    pass


