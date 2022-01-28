""" This module allows the user to select an algorithm to run"""

import logging
from steinerpy.algorithms.sstar import SstarMM0UN

from .algorithms import SstarHS, Kruskal, SstarBS, SstarMM, SstarMM0, \
                        SstarHSUN, SstarBSUN, SstarMMUN, \
                        SstarMMLP, SstarHSLP,\
                        SstarMMUNLP, SstarHSUNLP

from steinerpy.library.graphs.graph import GraphFactory

my_logger = logging.getLogger(__name__)

class Context:
    """ The Context class is responsible for passing a user's request to run a specific algorithm

    All results are stored in table (dict), which can easily be used to return results
    to the user.

    Parameters:
        terminals (list): A list of tuples representing terminals on a graph. 
            Exact format depends on the type of graph used (see below).
        graph (SquareGrid, MyGraph): Graph classes from superclass IGraph.
            Created using 'GraphFactory' class from the 'graph' module
        instances (dict): Stores the results of a queried algorithm
        
    Todo: 
        * rename some attributes?
        
    .. _Google Python Style Guide:
       http://google.github.io/styleguide/pyguide.html

    """
    def __init__(self, graph, terminals):
        self._graph = graph
        self._terminals = terminals
        self.instances = {}

    @property
    def strategy(self):
        """Getter method. The latest algorithm that was run"""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        """Setter method. Sets the latest algorithm to that was run """
        self._strategy = strategy

    def run(self, strategy, **kwargs):
        """Query an algorithm based on the user's selection

        Args:
            strategy (str): user's desired algorithm to run
        
        Returns:
            True: When no errors

        Raises:
            AssertionError: if strategy doesn't exist or has not been implemented yet

        """
        # Update strategy to most recent run
        self.strategy = strategy

        # run based on input
        if self.strategy == "S*-HS":
            self.instances[self.strategy] = SstarHS(self._graph, self._terminals)
        elif  self.strategy == "Kruskal":
            self.instances[self.strategy] = Kruskal(self._graph, self._terminals)
        elif self.strategy == "S*-BS":
            self.instances[self.strategy] = SstarBS(self._graph, self._terminals)
        elif self.strategy == "S*-MM":
            self.instances[self.strategy] = SstarMM(self._graph, self._terminals)
        elif self.strategy == "S*-MM0":
            self.instances[self.strategy] = SstarMM0(self._graph, self._terminals)
        elif self.strategy == "S*-HS-UN":
            self.instances[self.strategy] = SstarHSUN(self._graph, self._terminals)
        elif self.strategy == "S*-BS-UN":
            self.instances[self.strategy] = SstarBSUN(self._graph, self._terminals)
        elif self.strategy == "S*-MM-UN":
            self.instances[self.strategy] = SstarMMUN(self._graph, self._terminals)
        elif self.strategy == "S*-MM0-UN":
            self.instances[self.strategy] = SstarMM0UN(self._graph, self._terminals)                                
        elif self.strategy == "S*-MM-LP":
            self.instances[self.strategy] = SstarMMLP(self._graph, self._terminals)                                
        elif self.strategy == "S*-MM-UN-LP":
            self.instances[self.strategy] = SstarMMUNLP(self._graph, self._terminals)                                
        elif self.strategy == "S*-HS-LP":
            self.instances[self.strategy] = SstarHSLP(self._graph, self._terminals)                                
        elif self.strategy == "S*-HS-UN-LP":
            self.instances[self.strategy] = SstarHSUNLP(self._graph, self._terminals)                                

        else:
            raise ValueError("Strategy {} is not found".format(self.strategy))
        # Log run
        my_logger.info("Running {}".format(self.strategy))
        self.instances[self.strategy].run_algorithm(**kwargs)
        # Return True when no errors
        return True

    def return_solutions(self):
        """returns the results of the last run
        
        The user can also choose to set the class property `strategy`

        """ 
        # return latest strategy
        return self.instances[self.strategy].return_solutions()

    def return_data_structures(self):
        """Return gcosts, parent, children, and root data structures
            for each component. 

        """
        data_structures = {}
        for c, v in self.instances[self.strategy].comps.items():
            data_structures.update({
                c: {"gcosts": v.g, 
                    "parent": v.parent,
                    "children": v.children,
                    "root": v.root}
            })

        return data_structures