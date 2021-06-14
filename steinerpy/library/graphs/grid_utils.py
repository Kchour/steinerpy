""" Utility functions for grid-based graph """
import numpy as np

def filter_points(path, points, grid_dim):
    # for each point in 'points', return only points within boundaries
    x = points[:,0]
    y = points[:,1]
    thresh = points[ (grid_dim[0] <= x)*(x <= grid_dim[1])* (grid_dim[2] <= y)*(y <= grid_dim[3])]
    return thresh

def init_grid(grid_dim, grid_size, init_val):
    # Add 1 to even out world coordinates
    # Add np ceil to ensure actual grid size is not bigger than desired
    # since for some values, the grid dim wont divide evenly by grid size
    # xvals = int((grid_dim[1] - grid_dim[0] + 1) / grid_size)
    # yvals = int((grid_dim[3] - grid_dim[2] + 1) / grid_size)

    xvals = int((grid_dim[1] - grid_dim[0]) / grid_size + 1)
    yvals = int((grid_dim[3] - grid_dim[2]) / grid_size + 1)
    
    if init_val != 0:
        return init_val * np.ones((yvals, xvals))
    else:
        return np.zeros((yvals, xvals))

def get_index(x, y, grid_size, grid_dim):
    # Convert from world coordinates to index
    #indx = int((x - grid_dim[0])/grid_size)
    #indy = int((y - grid_dim[2])/grid_size)
    # Make sure x,y fall within the grid physical boundaries
    if np.any((grid_dim[0] <= x) *
              (x <= grid_dim[1]) == 0) and np.any((grid_dim[2] <= y) *
                                                  (y <= grid_dim[3]) == 0):
        raise NameError('(x,y) world coordinates must be within boundaries')
    indx = np.round((x - grid_dim[0]) / grid_size).astype(int)
    indy = np.round((y - grid_dim[2]) / grid_size).astype(int)
    return (indx, indy)


# A list of all the points in our grid
def mesh_grid_list(grid_dim, grid_size):
    # Make a mesh grid with the specified grid size/dim
    xvals = np.arange(grid_dim[0], grid_dim[1], grid_size)
    yvals = np.arange(grid_dim[2], grid_dim[3], grid_size)
    x, y = np.meshgrid(xvals, yvals)
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    return points

def get_world(indx, indy, grid_size, grid_dim):
    # Convert from index to world coordinates
    x = (indx) * grid_size + grid_dim[0] 
    y = (indy) * grid_size + grid_dim[2] 
    return (x, y)