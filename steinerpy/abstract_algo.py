"""A module with the class `AbstractAlgorithm` defined"""

# from abc import ABC, abstractmethod
from abc import abstractmethod
from steinerpy.library.misc.abc_utils import abstract_attribute, ABC as newABC
from steinerpy.library.graphs.graph import IGraph
from typing import List

class AbstractAlgorithm(newABC):
    """An abstract barebones superclass for each algorithm implementation.

    All algorithm implementations should inhereit :py:class:: AbstractAlgorithm.
    Do not instantiate this directly!

    Attributes:
        terminals (list): A list of tuples representing terminals on a graph. 
            Exact format depends on the type of graph used (see below).
        graph (SquareGrid, MyGraph): Graph classes from superclass IGraph.
            Created using 'GraphFactory' class from the 'graph' module
        S (dict): A dictionary containing information to output Steiner Tree
            'sol': is a list of tree edges, e.g. ((x1,y1),(x2,y2)) if using SquareGrid graph
            'dist': is a list of each tree edge's distance cost 
            'path': is a list of vertices of G, that make up each tree edge
            'stats': {'run_time': x, closed_nodes: y, open_nodes: z}
    """
    def __init__(self, G, T):
        self.terminals = T
        self.graph = G
        self.S = {'sol':[], 'dist':[], 'path':[], 'stats':{}}
        # self.FLAG_STATUS_completeTree = False

    def return_solutions(self):
        """Return solution set of final tree

        Returns:
            S (dict): A dictionary containing information to output Steiner Tree

        """     
        return self.S

    @abstractmethod
    def run_algorithm(self):
        """Queries the algorithm and populates solution set 'S'
        
        This is an abstract method, which must be explicitly defined
        in subclasses

        """
        pass