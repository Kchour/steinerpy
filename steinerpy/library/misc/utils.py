""" Unfinished collection of utility functions/classes 

Todo:
    Read png based occupancy grid map
"""

# import png
import pdb
import numpy as np
import collections
np.set_printoptions(suppress=True)
import copy #for deep copy purposes

import sys
np.set_printoptions(threshold=sys.maxsize)
             
# Limited DEEP COPY IMPLEMENTATION
_dispatcher = {}

def _copy_list(l, dispatch):
    ret = l.copy()
    for idx, item in enumerate(ret):
        cp = dispatch.get(type(item))
        if cp is not None:
            ret[idx] = cp(item, dispatch)
    return ret

def _copy_dict(d, dispatch):
    ret = d.copy()
    for key, value in ret.items():
        cp = dispatch.get(type(value))
        if cp is not None:
            ret[key] = cp(value, dispatch)

    return ret

_dispatcher[list] = _copy_list
_dispatcher[dict] = _copy_dict

def deepcopy(sth):
    cp = _dispatcher.get(type(sth))
    if cp is None:
        return sth
    else:
        return cp(sth, _dispatcher)       

# DEFINE GRAPH RELATED DATA STRUCTURES
from operator import itemgetter
''' usage
    my_dict = {x: x**2 for x in range(10)}

    itemgetter(1, 3, 2, 5)(my_dict)
    #>>> (1, 9, 4, 25)
'''

''' Info:   Use depth first search to find cycles in a graph
    
    Input:   - a MyGraph object
             - a starting node

'''
import random as rdm
''' wip in class to implement Disjoint Union method for cycle detection '''
class CycleDetection:
    def __init__(self, terminals, visualize=False):
        self.disjointSet = set(terminals)
    
    '''Give a list of edges '''
    def update(self):
        for s in self.disjointSet:
            print("wip")
        
'''
    Usage: To visualize the path in cycle_detection
'''
from matplotlib import animation
'''
    Inputs
    - nxG:  networkx graph
    - fig:  matplotlib fig
    - ax:   matplotlib ax
    - pos:  networkx pos  
'''

#def merge_adjacency_list(list1, list2):
''' Merge components of a search algorithm 
    Input:
    - 'components' Hash Table
    - Tuples c1 and c2 ()

    Function:
    - Merges the openlist, closed, list, linkedlist, and current node
    Breaks ties between two lists (min of the two)

    Returns:
    - a modified 'components' hash table
'''
def merge_components(components, c1, c2):
    # Merge terminal 
    newTerminals = augment([c1],[c2])

    # merge closed lists
    newCL = augment(components[c1]['results']['closedList'], components[c2]['results']['closedList'])
    newCL = {k:min(v) for k,v in newCL.items()}

    # merge frontier
    newF = augment(components[c1]['results']['frontier'], components[c2]['results']['frontier'])
    newF = {k:min(v) for k,v in newF.items()}

    # merge parentList
    newP = augment(components[c1]['results']['parentList'], components[c2]['results']['parentList'])
    
    print('wip')
    # set(c1_data['results'])
    # c1_data['results'][]


    # reuse one of the SA objects, delete the other

    print("wip")

''' General purpose union function for lists, dicts

    Input
    - c1, c2    both of which are the same type (either list or dict)

    Output
    - A 
'''
def augment(c1, c2):
    if isinstance(c1,list):
        return list(set(c1).union(set(c2)))
    elif isinstance(c1, dict):
        s1 = set(c1).union(set(c2))
        d1 = {}
        for s in s1:
            val = []
            if s in c1:
                val.append(c1[s])
            if s in c2:
                val.append(c2[s]) 
            d1[s] = val
        return d1
    elif isinstance(c1, set):
        return set(c1).union(set(c2))
    elif isinstance(c1, tuple):
        return tuple(set(c1).union(set(c2)))


class MyTimer:
    """Helper class to time functions and short scripts 

    To be used in conjunction with 'from timeit import default_timer as timer'

    Todo:
        * Create an initializer class method?

    """
    timeTable = {}

    @classmethod
    def add_time(cls, name, time):
        """Update time table
        
        Parameters:
            name (str): A user-defined string representing a function
            time (float): The time it took (s) to run said function

        """
        if name not in cls.timeTable:
            cls.timeTable[name] = [time]
        else:
            cls.timeTable[name].append(time)

    @classmethod
    def reset(cls):
        """IMPORTANT FUNCTION. Required to reinitialize variables"""
        cls.timeTable = {}

import sys
import time
class Progress:

    def __init__(self, num_of_iter):
        
        self.n = num_of_iter 
        self.i = 0
        self.bar_length = 21

    def next(self):
        # for i in range(self.n):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        # sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
        # incremental bar ===, length of bar, percentage 
        if self.n - 1 == 0:
            perc = 100
        else:
            perc = (100/(self.n-1)*self.i)
        sys.stdout.write("[{:{}}] {:.1f}%".format("="*int(perc*(self.bar_length)/100), self.bar_length, perc))
        sys.stdout.flush()
        self.i+=1
        # sleep(0.25)
        # new line
        # sys.stdout.write("\n")
    
    def finish(self):
        sys.stdout.write("\n")


if __name__ == "__main__":
    p = Progress(1)
    for i in range(1):
        p.next()
        time.sleep(0.25)
    
    p.finish()