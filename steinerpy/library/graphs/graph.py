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
from .grid_utils import init_grid
from .grid_utils import get_index
from .grid_utils import get_world
from .ogm import OccupancyGridMap
import matplotlib.pyplot as plt

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

    @abstractmethod
    def node_count(self):
        pass

    @abstractmethod
    def edge_count(self):
        pass
    
class GraphFactory(ABC):
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
    def create_graph(type_: str, **kwargs ) -> IGraph:
        try:
            if type_ is "SquareGrid":
                return SquareGrid(**kwargs)
            elif type_ is "Generic":
                return MyGraph(**kwargs)
            elif type_ is "SquareGridDepot":
                return SquareGridDepot(**kwargs)
            raise AssertionError("Graph type not defined")
        except AssertionError as _e:
            print(_e)
            raise

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
    def __init__(self, grid=None, grid_dim=None, grid_size=None, n_type=4, obstacles=None):
        self.obstacles = obstacles
        self.xwidth = grid_dim[1] - grid_dim[0]  + 1
        self.yheight = grid_dim[3] - grid_dim[2] + 1

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

        # self.node_count = np.floor(self.xwidth * self.yheight / grid_size)- 1 - self.xwidth - self.yheight
        m,n = np.floor(self.yheight / grid_size), np.floor(self.xwidth / grid_size)
        self._node_count = m*n
        if n_type == 4:
            self._edge_count = (m-1)*n + (n-1)*m
        else:
            # King's graph
            self._edge_count = 4*m*n - 3*(n+m) +2 

    def edge_count(self):
        return self._edge_count

    def node_count(self):
        return self._node_count 

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

        """
        d = {}
        # get max degree
        max_degree = None
        for v in self.get_nodes():
            degree = len(list(self.neighbors(v)))
            if max_degree is None or degree > max_degree:
                max_degree = degree
            d.update({v: degree})

        # keep all nodes with degree less than max_degree
        b_nodes = (vkey for vkey, vval in d.items() if vval < max_degree)

        return b_nodes



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
                minX,
                maxX,
                minY,
                maxY],
            cmap='Blues')
        plt.title("Occupancy Grid Map")
        plt.axis('scaled')  #equal is another one
        plt.grid()
        
        self.fig.canvas.restore_region(background)
        # Draw artists on helper objects
        self.ax.draw_artist(im)

        self.fig.canvas.blit(self.ax.bbox)

        # must call fig.canvas.flush_events() (called by pause internally)
        self.fig.canvas.flush_events()
        plt.pause(0.5)

class SquareGridDepot(SquareGrid):
    """Subclass of SquareGrid which explicitly handles the case of
        zero-cost edges between depots
    
    """
    def __init__(self, grid=None, grid_dim=None, grid_size=None, n_type=4, obstacles=None, depots=None):
        if depots is None:
            raise ValueError("keyword 'depots' is empty")

        super().__init__(grid, grid_dim, grid_size, n_type, obstacles)

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
    """A class for the most generic graph type

    Stores both an adjaceny list and cost table

    Parameters:
        edge_dict (dict): Contains all edges and the respective weight, i.e. {('v1', 'v2'): 5.0}
        graph_type (str): "undirected" or None
        visualize (str): Whether to use networkx to visualize results

    Attributes:
        adjList (dict): For each node, a list of adjacent nodes are given
        edge_dict (dict): For each edge, a weight is given

    Todo: 
        - fix visualization for larger graphs
        - Need further test the adjaceny list!
        - Need to update Adjaceny list, whenever cost table is updated!

    """
    def __init__(self, edge_dict=None, graph_type="undirected", visualize=False):
        self.adjList = {}
        self.edge_dict = copy.deepcopy(edge_dict) 
        if graph_type is "undirected":
            for key in edge_dict.keys():
                self.edge_dict[(key[1], key[0])] = edge_dict[key]

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

        self._node_count = len(self.adjList)
        self._edge_count = len(self.edge_dict)

    def edge_count(self):
        return self._edge_count

    def node_count(self):
        return self._node_count 

    def update(self, edge_dict):
        """ Add edges to our graph
        
        Parameter:
            edge_dict (dict): i.e. {('v1', 'v2'): 5.0}
        
        """
        self.edge_dict.update(edge_dict)
   
    def remove(self, edge_dict):
        """ Delete specific edges in our graph

        """
        for e in edge_dict.keys():
            #self.edge_dict.pop(e)  #slightly slower
            del self.edge_dict[e]

    def neighbors(self, v):
        """Return a list of neighbors of v
        
        Todo:
            - just return adjList[v]!
        
        """
        neighs = [n for n in self.adjList[v]]
        return neighs
    
    def cost(self, from_node, to_node):
        '''Returns edge weight '''
        weight = self.edge_dict[(from_node, to_node)]
        return weight
