import numba as nb
from numba import typed
import math
import numpy as np
import random

from .numba_search_utils import PriorityQueue2D, PriorityQueue3D
from steinerpy.library.graphs.graph_numba import RectGrid3D

class UniSearchMemLimitFast:
    """Optimized version of Uni-directional, memory-limited search
    
    Requires numba

    """
    total_expanded_nodes = 0
    def __init__(self, graph, start:tuple, goal:tuple):
        # # try to load numba if not already loaded
        # if "numba" in sys.modules:
        #     pass
        # elif (spec := importlib.util.find_spec("numba")) is not None:
        #     # import numba
        #     module = importlib.util.module_from_spec(spec)
        #     sys.modules["numba"] = module
        #     spec.loader.exec_module(module)
        # else:
        #     raise ImportError("Cannot find 'numba', it's not installed?!")
        self.graph = graph
        self.start = start
        # self.g = typed.Dict.empty(nb.types.UniTuple(nb.int64, len(start)), nb.types.float64)
        if len(start) == 2:
            # 3d graph
            # store cost-to-come, init start state
            self.g = np.full((graph.xwidth, graph.yheight), np.inf)
            self.frontier = PriorityQueue2D()
        elif len(start) == 3:
            # 3d graph
            # store cost-to-come, init start state
            self.g = np.full((graph.x_len, graph.y_len, graph.z_len), np.inf)
            self.frontier = PriorityQueue3D()
        self.g[start] = 0
        self.goal = goal
        
        # init the front
        self.frontier.put(start, 0) 

    @classmethod
    def update_expanded_nodes(cls):
        cls.total_expanded_nodes +=1
    
    @classmethod
    def reset(cls):
        """Reset all class variables """
        cls.total_expanded_nodes = 0

    @nb.njit(cache=True)
    def _search(pq, graph, g, goal):
        """
        Params:
            pq: priority queue
            g:  a numba dict containing both the closed and open set
            b:  boundary nodes
            p:  parent dict

        """ 
        iteration = 0 
        while pq:
            _, current = pq.pop()

            # early stopping
            if current in goal:
                goal = goal - set(current)
                if not goal:
                    break

            if iteration % 1e3 == 0:
                print("searched nodes: ", iteration)

            for n in graph.neighbors(current):
                g_next = g[current] + graph.cost(current, n)
                if g[n] == np.inf or g_next < g[n]:
                    g[n] = g_next
                    pq.put(n, g_next)

            iteration += 1
        return g, iteration

    def use_algorithm(self):
        # need to wrap each guy properly
        UniSearchMemLimitFast.total_expanded_nodes = 0
        g, iteration = UniSearchMemLimitFast._search(self.frontier, self.graph, self.g, self.goal)
        self.g = g
        # self.parent = p
        UniSearchMemLimitFast.total_expanded_nodes = iteration