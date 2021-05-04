
import numpy as np
from .grid_utils import init_grid
from .grid_utils import get_index

class OccupancyGridMap:
    ''' A class for adding regions in a plot
    grid_dim:  grid world dimensions [minX, maxX, minY, maxY]
    grid_size: grid resolution [m]
    add_region(verticies): specify ordered vertices of polygon
    returns: all the points lying within the polygonal region
    '''
    def __init__(self, grid_size, grid_dim, obs=None):
        """grid_res: grid resolution [m]
         grid_dim: grid dimension [m x m]"""

        self.grid_size = grid_size
        self.grid_dim = grid_dim
        # init grid
        #xvals = (grid_dim[1] - grid_dim[0])/grid_size
        #yvals = (grid_dim[3] - grid_dim[2])/grid_size
        #self.grid = np.zeros((yvals,xvals))
        self.grid = init_grid(grid_dim, grid_size, 0)
        self.obs = obs

        sz = np.shape(self.grid)
        self.xwidth = sz[1]
        self.yheight = sz[0]

        # if obs is non-empty
        if obs is not None:
            self.add_obstacles(self.grid, obs)

    def return_grid(self):
        print("Returning Grid")
        return self.grid

    def add_obstacles(self, grid, obs):
        print("Adding Obstacles")

        # convert to numpy if not already
        if 'numpy' not in str(type(obs)):
            obs = np.array(obs)
        #minX = self.grid_dim[0]
        #maxX = self.grid_dim[1]
        #minY = self.grid_dim[2]
        #maxY = self.grid_dim[3]
        obj_inds = get_index(obs[:, 0], obs[:, 1], self.grid_size, self.grid_dim)
        self.grid[obj_inds[1], obj_inds[0]] = 1.0
        
        # for o in obs:
        #        (indx, indy) = get_index(o[0], o[1], self.grid_size, self.grid_dim)
        #	#indx = int((o[0] - minX)/self.grid_size)
        #	#indy = int((o[1] - minY)/self.grid_size)
        #	self.grid[indy, indx] = 1.0
