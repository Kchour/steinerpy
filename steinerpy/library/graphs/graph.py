""" Custom graph class structures

classes:

    :GraphFactory: Primarily used to create graphs. Serves as the client interface

    :IGraph: The interface class responsible for actually created graph objects

    :SquareGrid: A 2D grid-based graph 
        
        :neighbors(v): given tuple (x,y), returns neighbors
        :cost(v1, v2): given two tuples (v1, v2), returns cost of edge

    :MyGraph: User can define a generic graph, by giving a edge list
        
        :neighbors(v): similar to above, but v is generic here
        :cost(v1,v2): similar to above, but v1,v2 is generic

Todo:
    - Create a 3D graph class
    
"""

from abc import ABC, abstractmethod
import numpy as np
import copy
import itertools as it

from steinerpy.library.graphs import graph
from .grid_utils import init_grid
from .grid_utils import get_index
from .grid_utils import get_world
from .ogm import OccupancyGridMap
import matplotlib.pyplot as plt

from mayavi.mlab import points3d
from mayavi import mlab

class IGraph(ABC):
    """ Create an abstract interface factory interface class    

    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def neighbors(self, v):
        pass

    @abstractmethod
    def cost(self, from_node, to_node):
        pass

    def node_count(self):
        pass

    def edge_count(self):
        pass

class GraphFactory:
    """ A factory class used to create graph objects
    
    Create a square grid

    Example:
        minX = -15			# [m]
        maxX = 15           
        minY = -15
        maxY = 15
        grid = None         # pre-existing 2d numpy array?
        grid_size = 1       # grid fineness[m]
        grid_dim = [minX, maxX, minY, maxY]
        n_type = 8           # neighbor type

        # Create a squareGrid using GraphFactory
        sq = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)      

    Create generic graph

    Example:
        # Define some edges
        edgeDict = {('v1','v2'): 1,
                    ('v2','v3'): 1,
                    ('v3','v4'): 1,
                    ('v4','v5'): 1,
                    ('v5','v6'): 1,
                    ('v6','v7'): 1,
                    ('v7','v8'): 1,
                    ('v8','v5'): 1}
        
        # Create a generic graph using factory method
        genG = GraphFactory.create_graph("Generic", edge_dict = edgeDict, graph_type = "undirected", visualize=False)

    """
    @staticmethod
    def create_graph(type_: str, *args, **kwargs ) -> IGraph:
        try:
            if type_ == "SquareGrid":
                return SquareGrid(*args, **kwargs)
            elif type_ == "Generic":
                return MyGraph(*args, **kwargs)
            elif type_ == "SquareGridDepot":
                return SquareGridDepot(*args, **kwargs)
            elif type_ == "SquareGrid3D":
                return SquareGrid3D(*args, **kwargs)
            raise AssertionError("Graph type not defined")
        except AssertionError as _e:
            print(_e)
            raise


class SquareGrid3D(IGraph):
    """a 3d based grid class using numpy array as an underlying data structure
    
    Assume a unit grid size for now and origin of (0,0,0)

    Add support to differentiate indices and world coordinates
    
    """
    # constants, the first is based on grid_size 
    vl = 1
    SQ3 = 1.7320508075688772
    SQ2 = 1.4142135623730951
    C1 = SQ3 - SQ2
    C2 = SQ2 - vl
    C3 = vl

    def __init__(self, grid_dim: list, grid_size: float, obstacles: list=None):
        # self.name = None
        self.type = "grid_3d"

        self.grid_size = grid_size
        self.grid_dim = grid_dim
        self.x_len = int((grid_dim[1] - grid_dim[0] + 1)/grid_size)
        self.y_len = int((grid_dim[3] - grid_dim[2] + 1)/grid_size)
        self.z_len = int((grid_dim[5] - grid_dim[4] + 1)/grid_size)

        # create a 3d np array
        self.grid = np.zeros((self.x_len, self.y_len, self.z_len))

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
        x,y,z = node
        xs, ys, zs = start
        # get vector to destinatio node
        dx, dy, dz = x - xs, y - ys, z - zs
        # check cardinal directions
        res = (self.grid[xs + dx][ys][zs] == 0 and\
                self.grid[xs][ys + dy][zs] == 0 and\
                self.grid[xs][ys][zs + dz] == 0)

        # check target location
        return self.grid[x][y][z] == 0 and res

    def neighbors(self, node:tuple):
        """There are at most 26 neighbors, subject to obstacles"""
        (x,y,z) = node
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

        # apply filter to remove nodes
        results = filter(lambda x: self.in_bounds(x), results) 
        results = filter(lambda x: self.not_obstacles(node, x), results) 

        return results

    def cost(self, from_node, to_node):
        """ based on voxel distance (octile distance generalized to 3d)"""
        x1, y1, z1 = from_node
        x2, y2, z2 = to_node
        dx, dy, dz = abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)
        dmax = max(dx, dy, dz)
        dmin = min(dx, dy, dz)
        dmid = dx + dy + dz - dmin - dmax 
        return SquareGrid3D.C1*dmin + SquareGrid3D.C2*dmid + SquareGrid3D.C3*dmax
    
    # def show_grid(self):
    #     """matplotlib is garbage with tons of data"""
    #     ax = plt.figure().add_subplot(projection='3d')
    #     ax.voxels(self.grid, facecolors="red", edgecolor='k')
    #     plt.show(block=False)
    @mlab.show
    def show_grid(self):
        xx, yy, zz = np.where(self.grid==1)
        mlab.points3d(xx,yy,zz, mode="cube")
        


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
            x,y,z = np.random.randint((min_x, min_y, min_z), (max_x, max_y, max_z))
            if self.grid[x,y,z] == 0:
                # add samples as tuples
                samples.add((x,y,z))
        
        return list(samples)


class SquareGrid(IGraph):
    """A grid based graph class. Physical coordinates origin (0,0) starts at the lower-left corner,
        while the index coordinate origin starts at the top-left 

    Parameters:
        grid: gotta remove this
        grid_dim (tuple): In the form of (minX, maxX, minY ,maxY) 
        grid_size (float or int): Discrete size of each grid block (assumed uniform)
        n_type (int): Number of neighbors for each node, 4 or 8
        obstacles (list): Each element is a tuple, representing the obstacles on the graph 

    Attributes:
        obstacles (list): A list of tuples, representing obstacles (x,y)
        xwidth (int): the width of the grid
        yheight (int): the height of the grid
        grid (numpy.ndarray): an numpy array grid
        grid_dim (tuple): The physical coord limits expressed as (minX, maxX, minY ,maxY) 
        grid_size (float or int): Discrete size of each grid block (assumed uniform)
        neighbor_type (int): Number of neighbors for each node, 4 or 8 

    Todo:
        - Allow for grid transformations?
        - Fix height and width

    """
    def __init__(self, grid=None, grid_dim=None, grid_size=None, n_type=4, obstacles=None, name=None, graph_type="undirected"):
        self.name = None

        self.obstacles = obstacles
        self.xwidth = grid_dim[1] - grid_dim[0]  + 1
        self.yheight = grid_dim[3] - grid_dim[2] + 1

        self.graph_type = graph_type

        # FIXME: the corrected width and height
        # self.xwidth = grid_dim[1] - grid_dim[0] +  1
        # self.yheight = grid_dim[3] - grid_dim[2] + 1

        self.n_type = n_type

        # if obstacles are defined, add ogm. Else init an empty grid
        if obstacles is not None and obstacles:
            ogm = OccupancyGridMap(grid_size, grid_dim, obstacles)
            self.grid = ogm.grid 
        else:
            self.grid = init_grid(grid_dim, grid_size, 0)
        
        self.grid_dim = grid_dim
        self.grid_size = grid_size
        self.neighbor_type = n_type

        # self._edge_count = None
        self._node_count = None
        # # self.node_count = np.floor(self.xwidth * self.yheight / grid_size)- 1 - self.xwidth - self.yheight
        m,n = np.floor(self.yheight / grid_size), np.floor(self.xwidth / grid_size)
        # self._node_count = m*n

        # edge count
        if n_type == 4:
            # 4 neighbors (north, east, south, west)
            self._edge_count = (m-1)*n + (n-1)*m
        else:
            # King's graph (north, ne, east, se, south, sw, west, nw)
            self._edge_count = 4*m*n - 3*(n+m) +2 

    def edge_count(self):
        """Not precise, this is an upper bound only!"""
        if self._edge_count is None:
            self._edge_count = len(list(self.get_edges()))
        return self._edge_count

    def node_count(self):
        return np.count_nonzero(self.grid==0) 
        # if self._node_count is None:
        #     self._node_count = len(list(self.get_nodes()))
        # return self._node_count 

    def get_nodes(self):
        # return ((self.grid_dim[0]+x*self.grid_size, self.grid_dim[2]+y*self.grid_size) for x in range(self.xwidth) for y in range(self.yheight))       
        y_ind, x_ind = np.where(self.grid<1)
        # get world
        wx, wy = get_world(x_ind, y_ind, self.grid_size, self.grid_dim)
        return ((x, y) for x,y in zip(wx,wy))

    def get_edges(self):
        """Returns a dict of edges and their weights
        """
        # d = defaultdict(lambda: np.inf)
        # INFO This uses less memory
        d = {}
        for from_node in self.get_nodes():
            for to_node in self.get_nodes():
                if from_node != to_node:
                    d.update({(from_node, to_node): np.inf})
                else:
                    d.update({(from_node, to_node): 0})

        t = [(from_node, to_node) for from_node in self.get_nodes() for to_node in self.neighbors(from_node)]
        f = lambda a: self.cost(a[0], a[1])
        u = map(f, t)
        add_d = {a:b for a,b in zip(t,u)}
        d.update(add_d)
      

        # INFO: This uses more memory
        # d = {(from_node, to_node): self.cost(from_node, to_node) for from_node in self.get_nodes() for to_node in self.neighbors(from_node)}
        return d

    def get_adjacency_matrix(self):
        nodes = list(self.get_nodes())
        edges = self.get_edges()

        n = len(nodes)
        grid = np.inf*np.ones((n,n))

        self.adj_map_to_wc = {}
        self.wc_map_to_adj = {}
        for i, a in enumerate(nodes):
            self.adj_map_to_wc[i] = a
            self.wc_map_to_adj[a] = i

        for e,v in edges.items():
            # c = np.array(list(map(lambda u: self.cost(u[0], u[1]), [(a,b) for b in nodes if b in self.neighbors(a)])))  
            i,j = self.wc_map_to_adj[e[0]], self.wc_map_to_adj[e[1]]          
            grid[i][j] = v

        print("")
        return grid

    def get_boundary_nodes(self):
        """Defined as nodes which have degree < max_degree

        WARNING: Does not work correctly

        """
        grad = np.gradient(self.grid)
        y_b = np.where(abs(grad[0])==0.5)
        x_b = np.where(abs(grad[1])==0.5)


        nodes = set((x1,x2) for x2, x1 in it.chain(zip(*x_b), zip(*y_b)))


        # d = {}
        # # get max degree
        # max_degree = None
        # for v in self.get_nodes():
        #     degree = len(list(self.neighbors(v)))
        #     if max_degree is None or degree > max_degree:
        #         max_degree = degree
        #     d.update({v: degree})

        # # keep all nodes with degree less than max_degree
        # b_nodes = (vkey for vkey, vval in d.items() if vval < max_degree)

        # return b_nodes
        return nodes

    def set_obstacles(self, obstacles):
        """ 
        parameter:
            obstacle (list): a list of tuples, where tuple is an obstacle (x,y)

        """
        ogm = OccupancyGridMap(self.grid_size, self.grid_dim, obstacles)
        self.obstacles = obstacles
        self.grid = ogm.grid 

    def in_bounds(self, ind, type_='map'):
        """ Test whether a coordinate is inside the grid boundaries

        Parameters:
            ind (tuple): x,y coordinate
            type_ (str): either map coordinates 'map', or physical world coordinates 'world'
        
        Returns:
            True: if coordinate is inside the boundaries
            False: otherwise

        """
        if type_ == 'world':
            (x, y) = ind
            return self.grid_dim[0] <= x <= self.grid_dim[1] and self.grid_dim[2] <= y <= self.grid_dim[3]
        else:
            # grid indices
            (indx, indy) = ind
            xcells = int(np.floor((self.xwidth) / self.grid_size))
            ycells = int(np.floor((self.yheight) / self.grid_size))
            return 0 <= indx <= xcells and 0 <= indy <= ycells

    def not_obstacles(self, ind, type_='map'):
        """ Test whether a coordinate coincides with an obstacle

        Parameters:
            ind (tuple): x,y coordinate
            type_ (str): either map coordinates 'map', or physical world coordinates 'world'
        
        Returns:
            True: if not an obstacle
            False: otherwise

        """
        if type_ == 'world':
            # convert world to ind first
            (indx, indy) = get_index(ind[0], ind[1], self.grid_size, self.grid_dim)
            return self.grid[indy, indx] == 0
        else:
            (indx, indy) = ind
            return self.grid[indy, indx] == 0
            

    def neighbors(self, node):
        """ Compute neighbors of a node, keep working in world coordinates 
      
        Parameter:
            node (tuple): node that we want to find neighbors about

        Returns:
            results (iterable): an iterable generator of neighbors
            
        """
        (x, y) = node
        if self.neighbor_type == 4:
            results = [(x + self.grid_size, y), (x, y - self.grid_size),
                       (x - self.grid_size, y), (x, y + self.grid_size)]
        elif self.neighbor_type == 8:
            results = [(x + self.grid_size, y), (x, y - self.grid_size),
                       (x - self.grid_size, y), (x, y + self.grid_size),
                       (x + self.grid_size, y + self.grid_size), (x + self.grid_size, y - self.grid_size),
                       (x - self.grid_size, y - self.grid_size), (x - self.grid_size, y + self.grid_size)]

        # Only return coordinates that are in range
        results = filter(lambda x: self.in_bounds(x, type_='world'), results)

        # Only return coordinates that are not obstacles
        results = filter(lambda x: self.not_obstacles(x, type_='world'), results)

        return results

    # Cost of moving from one node to another (edge cost)
    def cost(self, from_node, to_node):
        """edge cost based on grid-based calculations """
        (x1, y1) = from_node
        (x2, y2) = to_node
        dmax = max(abs(x1 - x2), abs(y1 - y2))
        dmin = min(abs(x1 - x2), abs(y1 - y2))
        return 1.414*dmin + (dmax - dmin)
        # return 1.41421*dmin + (dmax - dmin)
        # # 1.4142135623730951

        # a = from_node
        # b = to_node
        # v = (b[0] - a[0], b[1] - a[1])
        # return np.hypot(v[0], v[1])

        # (x1, y1) = from_node
        # (x2, y2) = to_node
        # return abs(x1 - x2) + abs(y1 - y2)
        # return 1

    def show_grid(self):
        """A method to display the current occupancy grid"""

        # Get grid dims
        minX, maxX, minY, maxY = self.grid_dim

        # get fig, ax objects
        self.fig = plt.figure()
        self.ax = self.fig.gca()

        # plotting in a non-blocking manner
        # plt.ion()
        # plt.draw()

        plt.pause(0.75)

        self.fig.canvas.draw_idle() 
        plt.show(block=False)

        # if interacting with canvas directly, must flush events after drawing
        # self.fig.canvas.flush_events()
        background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        # show grid as an image i.e. on a 2d raster
        # cmap _r indicates reversed (i.e. Blues_r, Black_r)
        # plt.draw()
        im = self.ax.imshow(
            self.grid,
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

        xmin, xmax, ymin, ymax = self.grid_dim
        # self.ax.set_xticks(np.arange(xmin, xmax+1, self.grid_size))
        # self.ax.set_yticks(np.arange(ymin, ymax+1, self.grid_size))
        # self.ax.set_xticklabels(np.arange(xmin, xmax+1, self.grid_size))
        # self.ax.set_yticklabels(np.arange(ymin, ymax+1, self.grid_size))
        
        plt.title("Occupancy Grid Map")
        plt.axis('scaled')  #equal is another one
        # plt.grid()
        
        self.fig.canvas.restore_region(background)
        # Draw artists on helper objects
        self.ax.draw_artist(im)

        self.fig.canvas.blit(self.ax.bbox)

        # must call fig.canvas.flush_events() (called by pause internally)
        self.fig.canvas.flush_events()
        # plt.pause(0.5)

        return self.fig, self.ax

    def sample_uniform(self, num_of_samples: int):
        """uniformly sample the free space"""
        min_x, max_x, min_y, max_y = self.grid_dim
        samples = set()
        while len(samples) < min(num_of_samples, self.node_count()):
            x,y = np.random.randint((min_x, min_y), (max_x, max_y))
            if self.grid[y,x] == 0:
                # add samples as tuples
                samples.add((x,y))
        
        return list(samples)

class SquareGridDepot(SquareGrid):
    """Subclass of SquareGrid which explicitly handles the case of
        zero-cost edges between depots
    
    """
    def __init__(self, grid=None, grid_dim=None, grid_size=None, n_type=4, obstacles=None, depots=None, name=None, graph_type="undirected"):
        if depots is None:
            raise ValueError("keyword 'depots' is empty")

        super().__init__(grid, grid_dim, grid_size, n_type, obstacles, name=name, graph_type=graph_type)

        # Save depots
        self.depots = depots

    def neighbors(self, node):
        """ Compute neighbors of a node, keep working in world coordinates 
      
        Parameter:
            node (tuple): node that we want to find neighbors about

        Returns:
            results (iterable): an iterable generator of neighbors
            
        """
        (x, y) = node
        if self.neighbor_type == 4:
            results = [(x + self.grid_size, y), (x, y - self.grid_size),
                       (x - self.grid_size, y), (x, y + self.grid_size)]
        elif self.neighbor_type == 8:
            results = [(x + self.grid_size, y), (x, y - self.grid_size),
                       (x - self.grid_size, y), (x, y + self.grid_size),
                       (x + self.grid_size, y + self.grid_size), (x + self.grid_size, y - self.grid_size),
                       (x - self.grid_size, y - self.grid_size), (x - self.grid_size, y + self.grid_size)]

        # Add other depots if input node is a depot
        if node in self.depots:
            for d in self.depots:
                if d != node:
                    results.append((d))

        # Only return coordinates that are in range
        results = filter(lambda x: self.in_bounds(x, type_='world'), results)

        # Only return coordinates that are not obstacles
        results = filter(lambda x: self.not_obstacles(x, type_='world'), results)

        return results

    def cost(self, from_node, to_node):
        """edge cost based on grid-based calculations """
        # Add zero-cost edges between depots
        if from_node in self.depots and to_node in self.depots:
            return 0
        else:        
            (x1, y1) = from_node
            (x2, y2) = to_node
            dmax = max(abs(x1 - x2), abs(y1 - y2))
            dmin = min(abs(x1 - x2), abs(y1 - y2))
            return 1.414*dmin + (dmax - dmin)
            # return 1.41421*dmin + (dmax - dmin)
            # # 1.4142135623730951

            # a = from_node
            # b = to_node
            # v = (b[0] - a[0], b[1] - a[1])
            # return np.hypot(v[0], v[1])

            # (x1, y1) = from_node
            # (x2, y2) = to_node
            # return abs(x1 - x2) + abs(y1 - y2)
            # return 1

class MyGraph(IGraph):
    """A class for the most generic graph type. Stores both an adjaceny list and cost table
    Parameters:
        edge_dict (dict): Contains all edges and the respective weight, i.e. {('v1', 'v2'): 5.0}. Weights can be anything (a list, a dict)
        vertex_dict (dict): Contains a key-value pair of vertices and their weights
        graph_type (str): "undirected" or "directed" (by default "directed")
        visualize (str): Whether to visualize results using matplotlib (not implemented yet)
        deep_copy (bool): Whether to create a deep copy (rather than a shallow copy) of input dictionaries
    Attributes:
        adjList (dict): For each node, a list of adjacent nodes are given
        edge_dict (dict): For each edge, a weight is given
        vertex_dict (dict): For each node, a weight is given
    Todo: 
        - Add visualization capabilities
    """
    @staticmethod
    def mat_to_dict(mat):
        """Return a edge_dict representation of edges
        Parameter:
            mat (list of list): elements that are nonzero or non-None type specify an adjacency
        
        Return:
            edge_dict (dict): A dict where each entry is a edge, keyed by value of that edge (which could be a hyper edge)
        """
        edge_dict = {}
        for i, row_vect in enumerate(mat):
            for j, val in enumerate(row_vect):
                if val != 0 and val != None:
                    edge_dict.update({(i,j): val})
        return edge_dict

    def __init__(self, edge_dict=None, vertex_dict=None, graph_type="directed", deep_copy=True, graph_name=None, visualize=False):
        """Visualize not implemented
        """
        self.adjList = {}
        self.graph_type = graph_type
        self.graph_name=None

        # Deep copy to avoid modifying original graph by reference
        if deep_copy:
            self.edge_dict = copy.deepcopy(edge_dict) 
            self.vertex_dict = copy.deepcopy(vertex_dict)
        else:
            self.edge_dict = edge_dict
            self.vertex_dict = vertex_dict

        if edge_dict is None:
            self.edge_dict = {}
        if vertex_dict is None:
            self.vertex_dict = {}

        if self.edge_dict is not None:
            self._update_adj_list()
            
    def _update_adj_list(self):
        # undirected vs directed edges
        if self.graph_type == "undirected":
            temp = {}
            for key in self.edge_dict.keys():
                temp.update({(key[1], key[0]): self.edge_dict[key]})
            self.edge_dict.update(temp)
        
            # create an adjacency list!
            for key in self.edge_dict.keys():
                if key[0] not in self.adjList:
                    self.adjList[key[0]] = [key[1]]
                if key[1] not in self.adjList:
                    self.adjList[key[1]] = [key[0]]
                else:
                    if key[0] not in self.adjList[key[1]]:
                        self.adjList[key[1]].append(key[0])
                    if key[1] not in self.adjList[key[0]]:
                        self.adjList[key[0]].append(key[1])       
        else:
            for key in self.edge_dict.keys():
                if key[0] not in self.adjList:
                    self.adjList[key[0]] = [key[1]]
                else:
                    if key[1] not in self.adjList[key[0]]:
                        self.adjList[key[0]].append(key[1])
                if key[1] not in self.adjList:
                    self.adjList[key[1]] = []    

        # CONSIDER DELETING EMPTY ADJACENCY KEYS

        # Update self.vertex_dict
        if self.vertex_dict is not None:
            for v in self.adjList:
                if v not in self.vertex_dict:
                    self.vertex_dict[v] = None

    # CREATE SETTER AND GETTER FUNCTION FOR vertex_dict attribute
    # LET adjList keep track of vertex_dict instead! j

    @property
    def edges(self):
        return self.edge_dict

    @property 
    def vertices(self):
        return self.vertex_dict

    def edge_count(self):
        if self.edge_dict is not None:   
            return len(self.edge_dict)
        else:
            return 0

    def node_count(self):
        # adjlist can be zero here
        return max(len(self.adjList), len(self.vertex_dict))

    def get_vertices(self):
        return list(self.vertex_dict)

    def add_edge(self, edge_dict):
        """Add edges to our graph. Will silently replace edges if it already exists. 
        
        Arg:
            edge_dict (dict): i.e. {('v1', 'v2'): 5.0}
        
        """
        self.edge_dict.update(edge_dict)
        self._update_adj_list()

    def add_vertex(self, vertex_dict):
        """Vertex to our graph
        Arg:
            vertex_dict (dict): {'v1': 5}
        
        """
        self.vertex_dict.update(vertex_dict)
        # SHOULD WE ALSO UPDATE ADJ LIST TO CONTAIN NEW NODES?
   
    def remove_edges(self, edge_list):
        """Delete specific edges in our graph
        Arg:
            edge_list (iter of edges): 
        """
        for e in edge_list:
            
            # update adjacency list and edge dict
            if self.graph_type == "directed":
                # update adjacency list
                if e[1] in self.adjList[e[0]]:
                    self.adjList[e[0]].remove(e[1])
                
                # update edge_dict 
                # self.edge_dict.pop(e)  #slightly slower
                del self.edge_dict[e]   #raises KeyError if not there
            else:
                # update adjacency list in both directions
                if e[1] in self.adjList[e[0]]:
                    self.adjList[e[0]].remove(e[1])
                if e[0] in self.adjList[e[1]]:
                    self.adjList[e[1]].remove(e[0])
                
                # update edge dicts in both directions
                del self.edge_dict[e]
                del self.edge_dict[(e[1], e[0])]

    def remove_vertices(self, vertex_list):
        """Delete vertices from our graph
        Arg:
            vertex_list (iter of vertices):
        Todo:
            - delete edges associated with deleted vertex
        
        """
        for v in vertex_list:
            del self.vertex_dict[v]

    def neighbors(self, v):
        """Return a list of neighbors of v
        
        Todo:
            - just return adjList[v]!
        
        """
        # neighs = [n for n in self.adjList[v]]
        if v in self.adjList:
            neighs = self.adjList[v]
            return neighs
        else:
            return []
    
    def cost(self, *args, **kwargs):
        """Return cost of edge or vertex based on number of args
        Args:
            from_node, to_node (vertex, vertex): Key of vertices, contained in edgeDict
            node (vertex): Key of a single vertex, contained in vertexDict
        Kwargs:
            name (str): To determine the specific reference for multi-weighted edges of graphs
        """
        # unpack name if in kwargs
        try:
            # case 1) gets an edge weight. case 2) gets a vertex weight
            if len(args) == 2:
                from_node, to_node = args
                if "name" in kwargs:
                    name = kwargs["name"]

                    # if a list of names, then iterate through the list
                    if type(name)==list:
                        weight = self.edge_dict[(from_node, to_node)]
                        for n in name:
                            weight = weight[n]     
                    else:
                        weight = self.edge_dict[(from_node, to_node)][name]               
                    return weight
                weight = self.edge_dict[(from_node, to_node)]
                return weight
            elif len(args) == 1:
                node = args[0]
                if "name" in kwargs:
                    name = kwargs["name"]
                    
                    # if a list of names, then iterate through the list
                    if type(name)==list:
                        weight = self.vertex_dict[node]
                        for n in name:
                            weight = weight[n]     
                    else:
                        weight = self.vertex_dict[node][name]
                    return weight

                weight = self.vertex_dict[node]
                return weight
            else:
                print("vertex or edge not found using arguments: ", args)
                raise KeyError("vertex or edge not found using arguments: ")
        except Exception as E_:
            raise E_