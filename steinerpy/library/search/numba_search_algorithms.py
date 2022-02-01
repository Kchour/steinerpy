import numba as nb
from numba import typed
import math
import numpy as np
import random

from steinerpy.library.search.search_algorithms import UniSearchMemLimit
from .numba_search_utils import PriorityQueue
from steinerpy.library.graphs.graph_numba import RectGrid3D

class UniSearchMemLimit3DOpt:
    """Optimized version of Uni-directional, memory-limited search
    
    Requires numba

    """
    total_expanded_nodes = 0
    def __init__(self, graph, start:tuple, goal:tuple, memory_limit=math.inf):
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
        self.start = start
        self.pq = PriorityQueue()
        self.g = typed.Dict.empty(nb.types.UniTuple(nb.int64, len(start)), nb.types.float64)
        self.goal = goal
        self.memory_limit = memory_limit
        self.parent = typed.Dict.empty(nb.types.UniTuple(nb.int64, len(start)), nb.types.UniTuple(nb.int64, len(start)))
        self.boundary = typed.Dict.empty(nb.types.UniTuple(nb.int64, len(start)), nb.types.int64)
        self.graph = graph
        
        # init the front
        self.pq.put(start, 0) 

    @classmethod
    def update_expanded_nodes(cls):
        cls.total_expanded_nodes +=1
    
    @classmethod
    def reset(cls):
        """Reset all class variables """
        cls.total_expanded_nodes = 0

    @nb.njit
    def _search(pq, graph, g, b, p, memory_limit):
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
            # keep track of boundary nodes
            b[current] = 0

            # delete any parents that have no remaining frontier children
            if p[current] is not None:
                b[p[current]] -= 1
                if b[p[current]] <= 0:
                    del b[p[current]]

            if len(g)>2*memory_limit:
                    nodes = random.sample(g.keys(), k=int(memory_limit))
                    for node in nodes:
                        if node not in b and node not in pq:
                            del g[node] 
                            del p[node]

            for n in graph.neighbors(current):
                g_next = g[current] + graph.cost(current, n)
                if n not in g or g_next < g[n]:
                    g[n] = g_next
                    pq.put(n, g_next)
                    p[n] = current

                    # increment frontier children counter
                    b[current] += 1

            iteration += 1
        return g, p, iteration

    def use_algorithm(self):
        # need to wrap each guy properly
        UniSearchMemLimit3DOpt.total_expanded_nodes = 0
        g, p, iteration = UniSearchMemLimit3DOpt._search(self.pq, self.graph, self.g, self.boundary, self.parent, self.memory_limit)
        self.g = g
        self.parent = p
        UniSearchMemLimit3DOpt.total_expanded_nodes = iteration