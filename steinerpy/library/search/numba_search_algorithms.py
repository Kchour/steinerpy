import numba as nb
import math

from steinerpy.library.search.search_algorithms import UniSearchMemLimit
from .numba_search_utils import PriorityQueue
from steinerpy.library.graphs.graph_numba import RectGrid3D

class UniSearchMemLimitOpt:
    """Optimized version of Uni-directional, memory-limited search
    
    Requires numba 
    """
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
        self.pq = PriorityQueue()
        self.g = {start: 0}
        self.goal = goal
        self.memory_limit = memory_limit
        self.parent = {start: None}
        self.boundary = {} 

    @nb.njit
    def _search(pq, g, b, p):
        """
        Params:
            pq: priority queue
            g:  a numba dict containing both the closed and open set
            b:  boundary nodes
            p:  parent dict

        """   
        while pq:
            _, current = pq.pop()
            # keep track of boundary nodes
            b[current] = 0

            # delete any parents that have no remaining frontier children
            if p[current] is not None:
                b[p[current]] -= 1
                if b[p[current]] <= 0:
                    del b[p[current]]

            for n in g.neighbors(current):
                g_next = g[current] + g.cost(current, n)
                if n not in g or g_next < g[n]:
                    g[n] = g_next
                    pq.put(n, g_next)
                    p[n] = current

                    # increment frontier children counter
                    b[current] += 1

    def use_algorithm(self):
        UniSearchMemLimitOpt._search(self.pq, self.g, self.boundary, self.parent)