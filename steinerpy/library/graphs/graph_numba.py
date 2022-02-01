import numba as nb
from numba import typed, typeof
from numba.experimental import jitclass
import numpy as np
import logging

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

spec = [("type", nb.types.unicode_type),
        ('grid_size', nb.types.float64),
        ('grid_dim', typeof(list_instance)),
        ('x_len', nb.types.int16),
        ('y_len', nb.types.int16),
        ('z_len', nb.types.int16),
        ('coord_dims', nb.types.UniTuple(nb.types.int16, 3)),
        ('grid', nb.types.float64[:,:,:])
]

@jitclass(spec)
class RectGrid3D:
    """a 3d based grid class using numpy array as an underlying data structure
    
    Assume a unit grid size for now and origin of (0,0,0)

    Add support to differentiate indices and world coordinates
    
    """


    def __init__(self, grid_dim: list, grid_size: float, obstacles: list=None):
        # self.name = None
        self.type = "grid_3d"

        self.grid_size = grid_size
        self.grid_dim = grid_dim
        self.x_len = int((grid_dim[1] - grid_dim[0] + 1)/grid_size)
        self.y_len = int((grid_dim[3] - grid_dim[2] + 1)/grid_size)
        self.z_len = int((grid_dim[5] - grid_dim[4] + 1)/grid_size)

        # create a 3d np array
        self.coord_dims = (self.x_len, self.y_len, self.z_len)
        self.grid = np.zeros(self.coord_dims, dtype=np.float64)

        # set cells to 1 if obstacle. no need to store obstacles list
        # self.obstacles = obstacles 
        if obstacles is not None:
            for o in obstacles:
                x,y,z = o
                self.grid[x][y][z] = 1

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
        min_x, max_x, min_y, max_y, min_z, max_z = self.grid_dim
        samples = set()
        while len(samples) < min(num_of_samples, self.node_count()):
            # blah = np.random.randint((min_x, min_y, min_z), (max_x, max_y, max_z))
            x = np.random.randint(min_x, max_x)
            y = np.random.randint(min_y, max_y)
            z = np.random.randint(min_z, max_z)
            if self.grid[x,y,z] == 0:
                # add samples as tuples
                samples.add((x,y,z))
        
        return list(samples)

def grid3d_wrap(grid_dim, grid_size, obstacles):

    for i, v in enumerate(grid_dim):
        list_instance[i] = v
    # convert obstacles to typed list
    obs = nb.typed.List(obstacles)

    return RectGrid3D(list_instance, grid_size, obs)

if __name__ == "__main__":
    "try to create"
    for i, v in enumerate([0, 99, 0, 99, 0, 99]):
        list_instance[i] = v


    obs = nb.typed.List([(0,0,0)])
    ro = RectGrid3D(list_instance, 1, obs) 
    pass