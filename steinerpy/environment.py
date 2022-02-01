"""Interface for loading predefined environments"""

import os
import steinerpy.config as cfg
from steinerpy.library.graphs.graph import IGraph
from steinerpy.library.graphs.parser import DataParser

from .env_type import EnvType

class EnvLoader:

    @classmethod
    def load(cls, env_type: str, *args, **kwargs):
        """return a predefined environment based on env_type

        Params:
            mapf maps, pass in the map name as a string, i.e. 'XXXX.map'
            steinlib, pass in the relative path as a string, i.e. "B/XXXX.stp"
            grid_2d: pass in the relative path as a string, i.e. 'sc/XXXX.map'
            hanoi: pass in the integer number of disks 'n' and towers 'k'
            pancake: pass in the integer number of pancakes


            TODO: Not finished loading the rest yet

        """
        if env_type == EnvType.MAPF:
            # pass the map name
            return cls._mapf_loader(args[0])
        elif env_type == EnvType.STEINLIB:
            # pass the relative path
            return cls._steinlib_loader(args[0])
        elif env_type == EnvType.GRID_3D:
            return cls._grid3d_loader(args[0])
        elif env_type == EnvType.GRID_2D:
            return cls._grid2d_loader(args[0])

    @classmethod
    def _grid2d_loader(cls, map_rel_name: str)->IGraph:
        path = os.path.join(cfg.data_dir, "grid_2d", map_rel_name)
        return DataParser.parse(path, dataset_type="grid_2d")

    @classmethod
    def _grid3d_loader(cls, map: str)->IGraph:
        path = os.path.join(cfg.data_dir, "grid_3d", map)
        return DataParser.parse(path, dataset_type="grid_3d")

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
        """tower of hanoi graph
        
        n: number of disks (1,...,n)
        k: number of towers (1,...,k)
        
        """
        return HanoiGraph(n,k)

###########################################################################
# Implement structured graphs here and define how they can be loaded above
# Graphs must have the neighborhood and cost function defined
# as in IGraph
###########################################################################

class HanoiGraph(IGraph):
    """Tower of Hanoi for n-discs and k-towers. The objective is to move
    a set of discs from one stack (tower) to another, but the following rules
    must be obyed:

        1) only one disc can be moved per turn, which is the top most of each stack.
        2) The top-disc can only be moved to another stack which contains a larger top disc
            or is empty.

    Note that the top disc must be the smallest disc in a particular stack. Also we must have k>=3

    Notation: (disc3 loc., disc2 loc., disc1 loc)

    E.g. for (n,k)=(3,4) or 3 discs and 4 towers. Let 1,2,3,4 represent each tower, and each disc
    from largest to smallest 3,2,1. Then the tuple 3-(i,j,k) represents the location of disc 3,2,1 
    at tower 1<=i, j, k<=4 respectively.

    """
    def __init__(self, n, k):
        """k towers, n discs, usually represented as H^n_k
        """
        self.n = n
        self.k = k
        self.discs = list(range(n))
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
            # make sure all disc labels are at most k (tower)
            assert all([i<=self.k for i in v])
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
                    # or the destination tower's smallest disc is larger
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

class PancakeGraph(IGraph):
    """Pancake flipping graph (P_N), where N is the number of pancakes

    A stack of pancakes is given, which may be in some arbitrary configuration. The goal is to 
    flip the pancakes using a spatula so that the pancakes is sorted from largest (bottom) to smallest 
    (smallest).

    Given an n-tuple, (i1,...,in), let the index correspond to a row (top to bottom, left to right), while the actual
    value at the index is the pancake size.

    As an example, the desired solution for P_10 is (1, 2, 3, ..., 10)

    """
    def __init__(self, n):
        self.n = n

    def neighbors(self, v: tuple, debug=False):
        if debug:
            assert all([i<=self.n for i in v])
        neighs = []
        for ndx, i in enumerate(v):
            if ndx == 0:
                continue
            neighs.append(v[0:ndx+1][::-1]+v[ndx+1::])
        return neighs

    def cost(*args):
        """Edge costs should be the number of pancakes above flip point"""
        return 1

if __name__ == "__main__":
    # test hanoi graph with (n=3, k=3)
    # n discs, k towers 
    hanoi = HanoiGraph(3, 4)
    # check neighbors
    # print(hanoi.neighbors((1,2,1,1), debug=True))
    print(hanoi.neighbors((1,2,1), debug=True))

    pancake = PancakeGraph(4)
    # index corresponds to the pancake size, while the value
    # is the row
    print(pancake.neighbors((1,2,3,4)))
