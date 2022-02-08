import numba as nb
from numba import typed
import math
import numpy as np
import random
from typing import Iterable

from steinerpy.library.search.search_algorithms import UniSearchMemLimit

from .numba_search_utils import PriorityQueue2D, PriorityQueue3D, PriorityQueueArb
# from steinerpy.library.graphs.graph_numba import RectGrid3D

class UniSearchMemLimitFast:
    """Optimized version of Uni-directional, memory-limited search
    
    Requires numba

    """
    total_expanded_nodes = 0
    def __init__(self, graph, start:tuple, goal:Iterable[tuple]):
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

        if len(start) == 2 and type(start)==tuple:
            # 2d graph
            # store cost-to-come, init start state
            self.g = np.full((graph.xwidth, graph.yheight), np.inf)
            self.frontier = PriorityQueue2D()
        elif len(start) == 3 and type(start)==tuple:
            # 3d graph
            # store cost-to-come, init start state
            self.g = np.full((graph.x_len, graph.y_len, graph.z_len), np.inf)
            self.frontier = PriorityQueue3D()
        elif type(start) == str:
            # allow for arbitrary key types 
            self.g = nb.typed.Dict.empty(nb.types.unicode_type, nb.types.float64)
            self.frontier = PriorityQueueArb()
        self.g[start] = 0.0

        if goal is not None:
            self.goal = goal.copy()
            print(list(self.goal)[20:25], end="")
        
        # init the front
        self.frontier.put(start, 0.0)


    @classmethod
    def update_expanded_nodes(cls):
        cls.total_expanded_nodes +=1
    
    @classmethod
    def reset(cls):
        """Reset all class variables """
        cls.total_expanded_nodes = 0


    @nb.njit(cache=True)
    def _search_np(pq, graph, g, goal):
        """ Assume g is a numpy array
        Params:
            pq: priority queue
            g:  a numba array containing both the closed and open set
            b:  boundary nodes
            p:  parent dict

        """ 
        iteration = 0 
        while not pq.empty():
            _, current = pq.pop()

            # early stopping
            if current in goal:
                # print(current)
                goal.remove(current)
                if len(goal)==0:
                    break

            if iteration % 1e6 == 0 and iteration > 1:
                print("searched nodes: ", iteration)

            for n in graph.neighbors(current):
                g_next = g[current] + graph.cost(current, n)
                if g[n] == np.inf or g_next < g[n]:
                    g[n] = g_next
                    pq.put(n, g_next)

            iteration += 1
        return g, iteration

    @nb.njit(cache=True)
    def _search_dict(pq, graph, g, goal):
        """ Assume g is a dict
        Params:
            pq: priority queue
            g:  a numba dict containing both the closed and open set
            b:  boundary nodes
            p:  parent dict

        """ 
        iteration = 0 
        while not pq.empty():
            _, current = pq.pop()

            # early stopping
            if current in goal:
                # print(current)
                goal.remove(current)
                if len(goal)==0:
                    break

            if iteration % 1e6 == 0 and iteration > 1:
                print("searched nodes: ", iteration)

            for n in graph.neighbors(current):
                g_next = g[current] + graph.cost(current, n)
                if n not in g or g_next < g[n]:
                    g[n] = g_next
                    pq.put(n, g_next)

            iteration += 1
        return g, iteration

    def use_algorithm(self):
        # need to wrap each guy properly
        UniSearchMemLimitFast.total_expanded_nodes = 0
        if type(self.start) == tuple:
            # search using numpy g
            g, iteration = UniSearchMemLimitFast._search_np(self.frontier, self.graph, self.g, self.goal)
        elif type(self.start) == str:
            # search using dict g
            g, iteration = UniSearchMemLimitFast._search_dict(self.frontier, self.graph, self.g, self.goal)
        # self.g = g
        # self.parent = p
        UniSearchMemLimitFast.total_expanded_nodes = iteration

class UniSearchMemLimitFastSC(UniSearchMemLimitFast):
    """Same as above except stopping criteria (mainly for cdh) is allowed
    
    WORK IN PROGRESS"""

    def __init__(self, graph, start, goals, stopping_critiera:callable, cdh_table, lb, ub, pivot_counter):
        super().__init__(graph, start, goals)

        # also 
        self.stopping_criteria = stopping_critiera
        self.cdh_table = cdh_table
        self.lb = lb
        self.ub = ub
        self.pivot_counter = pivot_counter

    @nb.njit(cache=True)
    def _search(pq, graph, g, sc, cdh_table, start, lb, ub, pivot_counter):
        """dijkstra with stopping condition"""
        iteration = 0 
        while not pq.empty():
            _, current = pq.pop()

            # early stopping
            if sc(g, current, cdh_table, start, lb, ub, pivot_counter):
                break

            for n in graph.neighbors(current):
                g_next = g[current] + graph.cost(current, n)
                if g[n] == np.inf or g_next < g[n]:
                    g[n] = g_next
                    pq.put(n, g_next)

            iteration += 1
        return g, iteration

    def use_algorithm(self):
        # need to wrap each guy properly
        UniSearchMemLimitFastSC.total_expanded_nodes = 0
        g, iteration = UniSearchMemLimitFastSC._search(self.frontier, self.graph, self.g, self.stopping_criteria, self.cdh_table, self.start, self.lb, self.ub, self.pivot_counter)
        # self.g = g
        # self.parent = p
        UniSearchMemLimitFastSC.total_expanded_nodes = iteration