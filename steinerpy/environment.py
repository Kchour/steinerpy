"""Interface for loading predefined environments"""

import enum
import os
from enum import Enum, auto
import steinerpy.config as cfg
from steinerpy.library.graphs.graph import IGraph
from steinerpy.library.graphs.parser import DataParser

class EnvType(Enum):
    MAPF = auto()
    STEINLIB = auto()
    HANOI = auto()
    PANCAKE = auto()

class EnvLoader:

    @classmethod
    def load(cls, env_type: str, *args, **kwargs):
        """return a predefined environment based on env_type

        """
        if env_type == EnvType.MAPF.name:
            # pass the map name
            return cls._mapf_loader(args[0])
        elif env_type == EnvType.STEINLIB:
            # pass the relative path
            return cls._steinlib_loader(args[0])



    @classmethod
    def _mapf_loader(cls, map: str)->IGraph:
        """Assumed to be mapf folder
        
        e.g. map (str): Berlin_1_256.map

        """
        path = os.path.join(cfg.data_dir, "mapf", map)
        return DataParser.parse(path, dataset_type="mapf")

    @classmethod
    def _steinlib_loader(cls, relative_path: str)->IGraph:
        """Assumed to be steinlib folder

        e.g. relative_path (str): B/b01.stp

        """
        path = os.path.join(cfg.data_dir, "steinlib", relative_path)
        return DataParser.parse(path, dataset_type="steinlib")
    
    @classmethod
    def _hanoi_loader(cls, n: int, k:int):
        pass

###########################################################################
# Implement structured graphs here and define how they can be loaded above
# Graphs must have the neighborhood and cost function defined
# as in IGraph
###########################################################################

class HanoiGraph(IGraph):
    """Tower of Hanoi for n-discs and k-towers. The objective is move
    a set of discs from one stack (tower) to another, but the following rules
    must be obyed:

        1) only one disc can be moved per turn, which is the top most of each stack.
        2) The top-disc can only be moved to another stack which contains a larger top disc
            or is empty.

    Note that the top disc must be the smallest disc in a particular stack. Also we must have k>=3

    Notation: (disc3 loc., disc2 loc., disc1 loc)

    E.g. for (n,k)=(3,3). Let 1,2,3 represent each tower, and each disc
    from largest to smallest 3,2,1. Then the tuple 3-(i,j,k) represents the location of disc 3,2,1 
    at tower i, j, k respectively.

    """
    def __init__(self, n, k):
        """n discs, k towers
        """
        self.n = n
        self.k = k
        self.discs = list(range(k))
        self.discs.reverse()        

    def neighbors(self, v: tuple, debug=False):
        """Given a configuration, we will return all neighbors of v

        v = (n-th disc loc., n-1th disc loc., .... , 1st disc loc.)

        NOTE: 
        - the possible vertices is n cartesian product of (1,...,n) 
        - Let the smallest disc be labeled as 1, the largest as n
        - There are k^n vertices

        """
        if debug:
            # make sure vertex tuple is not longer than defined
            assert(len(v))==self.n
            # make sure all disc labels are at most n
            assert all([i<=self.n for i in v])
        # empty tower
        tower = [[] for _ in range(self.k)]
        # store discs in different towers
        for ndx, x in enumerate(v):
            disc = self.n - ndx
            tower[x-1].append(disc)

        # now find all neighbors
        neighbors = []
        # loop through source tower
        for n1, k1 in enumerate(tower, 1):
            # loop through resulting tower after an action
            for n2, k2 in enumerate(tower, 1):
                # avoid self edges
                if n1 != n2 and len(k1)>0:
                    # an action is feasible if the destination tower is empty
                    # of the destination tower's smallest disc is larger
                    if len(k2) == 0 or k1[-1] < k2[-1]:
                        # able = [-1 for _ in range(self.n)]
                        # able = (nth disc loc, n-1 th disc loc, ..., 1)
                        able = list(v)
                        # # retain unmoved discs on source
                        # for x in k1[0:-1]:
                        #     able[self.n - x] = n1
                        # # retain unmoved discs on auxilary towers
                        # if k2:
                        #     able[self.n - k2[-1]] = n2
                        able[self.n - k1[-1]] = n2
                        neighbors.append(tuple(able))
        return neighbors
    
    def cost(*args):
        return 1

if __name__ == "__main__":
    # test hanoi graph with (n=3, k=3)
    hanoi = HanoiGraph(10, 4)
    # check neighbors
    print(hanoi.neighbors((1,2,1,1), debug=True))