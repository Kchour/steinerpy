"""Custom data structures for our search algorithms. Import methods are:

Classes:
    CustomQueue:
    Queue:
    PriorityQueue:
    PriorityQueueHeap:
    DoublyLinkedList:
    CycleDetection:

Functions:
    heuristics: grid-based heuristics atm (maybe redundant)
    reconstruct_path: 
    path_length:
    
Todo: 
    - Remove redundencies (`Common` class and heuristics)

"""


import collections 
import numpy as np

class CustomQueue:
    """Template class for defining custom Queue"""
    def __init__(self):
        ''' self.elements = '''
        pass
    
    def empty(self):
        ''' return len(self.elements) == 0 '''
        pass
    
    def put(self, x):
        ''' define stuff '''
        pass
    
    def get(self):
        ''' define stuff '''
        pass

class Queue:
    """FIFO data structure """
    def __init__(self):
        self.elements = collections.deque()

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.popleft()


class PriorityQueue:
    """ A O(n) minimum priorityQueue implementation """
    def __init__(self):
        self.elements = {}

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        self.elements[item] = priority

    def get(self):
        """ 
        returns:
            priority, item: Returns the 'minimum' item with its corresponding priority value

        todo:
            - refactor the dict requirements
        """
        # Iterate through dictionary to find the item with the best priority
        best_item, best_priority = None, None
        for item, priority in self.elements.items():
            if best_priority is None or priority < best_priority:
                best_item, best_priority = item, priority

        # Remove the best item from the OPEN LIST
        del self.elements[best_item]

        # return
        return best_priority, best_item

    def get_test(self):
        """Get min without deleting """
        # Iterate through dictionary to find the item with the best priority
        best_item, best_priority = None, None
        for item, priority in self.elements.items():
            if best_priority is None or priority < best_priority:
                best_item, best_priority = item, priority

        return best_priority, best_item
    
    # Only necessary for removing redundant items
    def delete(self, item):
        if item in self.elements:
            del self.elements[item]


import heapq
import itertools as it
class PriorityQueueHeap:
    """ A min Priority Queue O(log(n)) with heapq library
    
    Modified to break ties based on input order

    Attributes:
        pq (list): a heapified list of elements
        ecount (generator): unique sequence id number (incrementer)
        REMOVED (str): for lazy deletion feature, marked nodes get deleted
        entry_table (dict): key entries are of non-marked elements
        elements (dict): added for backwards compatibility purposes

    References:
        Thanks RedBlobGames
    
    """

    def __init__(self):
        self.pq = []
        self.ecount = it.count()
        self.REMOVED = "<removed-task>"
        self.entry_table = {}
        self.elements = self.entry_table

        self.min_item = None
        
    def __contains__(self, item):
        """Override 'in' membership check """
        return item in self.entry_table
    
    def empty(self):
        return len(self.entry_table) == 0
    
    def put(self, item, priority):
        """ tie-breaking is done based on ecount or update existing item's priority
        """
        # Set item/task to sentinel value so it doesn't get used
        if item in self.entry_table:
            self.delete(item)
        # define entry to consist of priority, counter, and item/task to be completed
        entry = [priority, next(self.ecount), item]
        heapq.heappush(self.pq, entry)
        self.entry_table[item] = entry
 
    def get(self):
        """ return the item with min priority value, specifically a tuple (value, item)
        """
        # return heapq.heappop(self.elements)
        # only return priority and item
        while self.pq:
            priority, _, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_table[task]
                return (priority, task)
        raise KeyError('pop from an empty priority queue')

    def get_test(self):
        """For backward compatibility purposes """
        return self.get_min()

    def get_min(self):
        """Return min item without removing it. purge "REMOVED items"
        """
        # use heapqueue property
        entry = self.pq[0]
        while entry is not None: 
            priority = entry[0]
            task = entry[-1]
            if task is not self.REMOVED:
                self.min_item = (priority, task)
                return (priority, task)
            else:
                # deletes the nodes marked for deletion
                heapq.heappop(self.pq)
            entry = self.pq[0]
        raise KeyError('pop from an empty priority queue')


    # Only necessary for removing redundant items
    def delete(self, item):
        """Lazy delete """
        # get and remove entry of the item from table 
        entry = self.entry_table.pop(item)
        # set entry to a sentinel "deleted" value so that it doesn't get returned
        entry[-1] = self.REMOVED

class DoublyLinkedList:
    """ Doubly linked list implementation using dicts

    functions:
        push: adds data to the front
        append: adds data at the end
        insert: adds data at some location predfine location

    """
    def __init__(self):
        self.head = None
        self.tail = None
        self.elements = {self.head :{'next': None, 'prev': None}}

    def push(self, node):
        ''' store node in front of the lists '''
        self.elements[node] = {'next': self.head, 'prev': None}
        self.elements[self.head]['prev'] = node

        self.head = node

        # initialize tail if not already
        if self.tail is None:
            self.tail = node

    def append(self, node):
        ''' store node at the end '''
        self.elements[node] = {'next': None, 'prev': self.tail}
        self.elements[self.tail]['next'] = node

        self.tail = node

        # initialize head if not already
        if self.head is None:
            self.head = node

    def insert(self, node, prev, next):
        self.elements[node] = {'next': next, 'prev': prev}
        self.elements[prev]['next'] = node
        self.elements[next]['prev'] = node

    def print_list(self, node, method):
        curr = node
        path = []
        while curr is not None:
            path.append(curr)

            next = self.elements[curr]['next']
            prev = self.elements[curr]['prev']
            
            print(prev, curr, next)

            if method == 'forward':  
                curr = next
            elif method == 'reverse':
                curr = prev
            else:
                raise Exception("Please select a method ('forward', 'reverse')") 
            

        print(path)
        #return path
                
""" post-processing utility functions """
def reconstruct_path(parents, start, goal, order='forward'):
    '''Given a linked list, we rebuild a path

    paremeters:
        parents: a singly-linked list using python dict
        start: a tuple (x,y) position
        goal: a tuple (x,y) position
        order: 'forward', or 'reverse' 

    '''
    current = goal
    path = []
    while current != start and current!= None:
        # Detect cycles and break out of them
        if current in path:
            # print("WARNING CYCLE DETECTED")
            break
        path.append(current)
        #  current = parents[current]
        current = parents.get(current, None)

    if start != None:
        path.append(start)
    if order == 'forward':
        path.reverse()

    return path

def path_length(path):
    """ Gives us a path length for square-grid based paths """
    # compute costs
    diff_ = np.diff(path, axis=0)
    hyp_ = np.hypot(diff_[:,0], diff_[:,1])
    dist_ = sum(hyp_)
    return dist_

def heuristic(self, a, b, type_='manhattan'):
    """ example square grid-based heuristic functions """
    (x1, y1) = a
    (x2, y2) = b
    if type_ == 'manhattan':
        return abs(x1 - x2) + abs(y1 - y2)
    elif type_ == 'euclidean':
        v = [x2 - x1, y2 - y1]
        return np.hypot(v[0], v[1])
    elif type_ == 'diagonal_uniform':
        return max(abs(x1 - x2), abs(y1 - y2))
    elif type_ == 'diagonal_nonuniform':
        dmax = max(abs(x1 - x2), abs(y1 - y2))
        dmin = min(abs(x1 - x2), abs(y1 - y2))
        # return 1.4142135623730951*dmin + (dmax - dmin)
        return 1.414*dmin + (dmax - dmin)
        
class CycleDetection:
    """Class used to check whether the next added edge will give a cycle on graph.
        Implements the disjoint set algorithm

    Attributes:
        indices: is iterable list of indices (singleton tuples) corresponding to nodes of a graph

    Todo:
        * Add cache to reduce computation amount (uses more memory though)
    """
    def __init__(self, indices):
        self.indices = indices          # disjoint sets used to test cycles
        self.parent_table = {}
    
    def add_edge(self, ind1, ind2, test=False):
        """function will add an edge to `indicies` if said edge doesn't form a cycle

        workflow:
            1: check to see if ind1+ind2 (tuple addition) is a subset of k-th set
                Yes -> CYCLE
                No- > continue
            2: check intersection, to find merge points
            3: perform merge!

        Parameters:
            ind1: singleton tuples, i.e. (i,) for some integer i 
            ind2: see above
            test: only test-add the edge, don't actually add it
            force: forcefully add the edge, even if it forms a cycle

        Returns:
            True: if added edge induces a cycle!
            False: otherwise, then update the disjoint sets
               
        """
        # Step 1: Check whether incoming edge belongs to a subset - cycle
        temp = []
        for k in self.indices:
            if set(ind1 + ind2).issubset(set(k)):
                # print("cycle detected!")
                return True

            # Step 2: Find subsets to merge
            if test is False:
                if set(ind1 + ind2).intersection(set(k)):
                    temp.append(k)
        
        if test is False:
            self.indices.remove(temp[0])
            self.indices.remove(temp[1])

            # Step 3: Merge!
            self.indices.append(temp[0]+temp[1])

            # update parent table
            for t in temp[0]+temp[1]:
                self.parent_table.update({(t,): temp[0]+temp[1]})
                self.parent_table.update({(t,): temp[0]+temp[1]})

        return False

