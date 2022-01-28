""" Unfinished collection of utility functions/classes 

Todo:
    Read png based occupancy grid map
"""

# import png
from multiprocessing import Value
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
        print("\n")

    def next(self):
        # for i in range(self.n):
        # sys.stdout.write('\r')
        print('\r', end="")

        # the exact output you're looking for:
        # sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
        # incremental bar ===, length of bar, percentage 
        if self.n - 1 == 0:
            perc = 100
        else:
            perc = (100/(self.n-1)*self.i)
        # sys.stdout.write("[{:{}}] {:.1f}%".format("="*int(perc*(self.bar_length)/100), self.bar_length, perc))
        # sys.stdout.flush()
        print("[{:{}}] {:.1f}%".format("="*int(perc*(self.bar_length)/100), self.bar_length, perc), end="")
        self.i+=1
        # sleep(0.25)
        # new line
        # sys.stdout.write("\n")
    
    def finish(self):
        # sys.stdout.write("\n")
        print("")

from collections import UserDict

class ChangingKeysDict(UserDict):
    """Class to keep track of changing keys which are 
        composed of base_keys

        TODO: Need to handle merging and when data keys contain other
        things besides base_keys

        TODO: specify whether directed or undirected

        Attributes:
            data (dict): inherited from UserDict. The actual dictionary.


    """
    def __init__(self, base_keys: list, key_type:str=None,*args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize all base keys
        # Use to keep track of all references to itself
        self.base_keys_ref = {}
        for k in base_keys:
            self.base_keys_ref[k] = set()

        if key_type is None:
            raise ValueError("key_type not specified!")
        elif key_type == "undirected" or key_type == "directed":
            self.key_type = key_type
        else:
            raise ValueError("key_type {} is not supported".format("".join(["\'",key_type,"\'"])))

    def __delitem__(self, key):
        """For each base_key in key, remove a reference to it

        """        
        for base_key in key:
            self.base_keys_ref[base_key].remove(key)

        super().__delitem__(key)

    def __setitem__(self, key: tuple, value: float):
        """For each base_key in key add a reference to it

        """
        for base_key in key:
            # try adding to base_key
            self.base_keys_ref[base_key].add(key)

        super().__setitem__(key, value)

    def change_keys(self, mapping: dict):
        """For each new base_key, scan all references and update
            their name

        """
        for base_key_old, base_key_new in mapping.items():
            # get data table references for old key
            refs = self.base_keys_ref[base_key_old]

            # add redefined base_key to ref table
            # dont initialize if new base key is already present.
            # else create a new one
            if base_key_new not in self.base_keys_ref:
                self.base_keys_ref[base_key_new] = set()            

            # scan over each ref, make a copy with a new key
            # delete the older one
            for ref in refs:
                # create a list
                list_ref = list(ref)

                # find index of old key
                ind = list_ref.index(base_key_old)

                # create a copy of ref with new key
                list_new_ref = list_ref[0:ind] + [base_key_new] + list_ref[ind+1:]

                # turn into tuple
                new_ref = tuple(list_new_ref)
                self.base_keys_ref[base_key_new].add(new_ref)

                # now update the data dict
                # if undirected, just take the minimum of the two values
                # maintain lexicographic ordering
                a,b = new_ref
                if self.key_type == "undirected":
                    if (b,a) in self.data:
                        min_val = min(self.data[(b,a)], self.data[ref]) 
                    else:
                        min_val = self.data[ref]    
                
                    if a > b:
                        a, b = b, a
                    
                    self.data[new_ref] = min_val
                else:
                    self.data[new_ref] = self.data[ref]
                
                # make sure other data references to old base key are changed
                for v in ref:
                    if v != base_key_old:
                        self.base_keys_ref[v].remove(ref)
                        self.base_keys_ref[v].add(new_ref)
                
                del self.data[ref]

            del self.base_keys_ref[base_key_old]
        
if __name__ == "__main__":


    # ckd = ChangingKeysDict(["a", "b", "c", "d"])
    # ckd.update({("a", "b"): 1, ("b", "c"): 2, ("c", "d"): 3})

    # # ckd.change_keys({"a": "apple"})
    # # ckd.change_keys({"b": "banana"})
    # # ckd.change_keys({"c": "cherry"})
    # # ckd.change_keys({"d": "deer"})

    # ckd.change_keys({"a": "apple", "b": "banana", "c": "cherry", "d": "deer"})

    # # try merging
    # ckd.change_keys({"apple": "sandwich", 'cherry': "sandwich"})

    p = Progress(1)
    for i in range(1):
        p.next()
        time.sleep(0.25)
    
    p.finish()