"""Numba accelerated priority queue based on lazy deletion. Implementation
    is a bit hacky because we are using a typed.List([Bool]) to
    keep track of entries in the pq list for deletions

    We also assume items in each entry are a tuple of integers 
"""
from typing import List, Dict, Tuple 
from heapq import heappush, heappop
import numba as nb
from numba.experimental import jitclass



# priority, counter, item, removed
entry_def = (0.0, 0, (0,0,0), nb.typed.List([False]))
entry_type = nb.typeof(entry_def)

# @jitclass
class PriorityQueue:
    pq: List[entry_type]
    entry_finder: Dict[Tuple[int, int], entry_type]
    counter: int
    entry: entry_type

    def __init__(self):
        # add an item to help numba infer type
        self.pq = nb.typed.List.empty_list( (0.0, 0, (0,0,0), nb.typed.List([False])) )
        self.entry_finder = nb.typed.Dict.empty( (0, 0, 0), (0.0, 0, (0,0,0), nb.typed.List([False])))
        self.counter = 0

    def put(self, item: Tuple[int, int], priority: float = 0.0):
        """Add a new item or update the priority of an existing item"""
        if item in self.entry_finder:
            # remove items
            self.remove_item(item)

        self.counter += 1
        # entry = Entry(priority, self.counter, item)
        entry = (priority, self.counter, item, nb.typed.List([False]))
        self.entry_finder[item] = entry
        # heappush(self.pq, entry)
        # heappush(self.pq, (priority, self.counter, item, False))
        heappush(self.pq, entry)

    def remove_item(self, item: Tuple[int, int]):
        """Mark an existing item as REMOVED.  Raise KeyError if not found."""
        self.entry = self.entry_finder.pop(item)
        self.entry[3][0] = True

    def pop(self):
        """Remove and return the lowest priority item. Raise KeyError if empty."""
        while self.pq:
            priority, count, item, removed = heappop(self.pq)
            # entry = heappop(self.pq)
            if not removed[0]:
                # del self.entry_finder[entry.item]
                del self.entry_finder[item]
                return priority, item
        raise KeyError("pop from an empty priority queue")

    def empty(self):
        return len(self.entry_finder)==0

# extend pure python class PriorityQueue (we can do this, but we cannot extend jitclasses)
@jitclass
class PriorityQueue3D(PriorityQueue):
    pq: List[entry_type]
    entry_finder: Dict[Tuple[int, int, int], entry_type]
    counter: int
    entry: entry_type

    def __init__(self):
        # add an item to help numba infer type
        self.pq = nb.typed.List.empty_list( (0.0, 0, (0,0,0), nb.typed.List([False])) )
        self.entry_finder = nb.typed.Dict.empty( (0, 0, 0), (0.0, 0, (0,0,0), nb.typed.List([False])))
        self.counter = 0

entry_def = (0.0, 0, (0,0), nb.typed.List([False]))
entry_type = nb.typeof(entry_def)
@jitclass
class PriorityQueue2D(PriorityQueue):
    pq: List[entry_type]
    entry_finder: Dict[Tuple[int, int], entry_type]
    counter: int
    entry: entry_type

    def __init__(self):
        # add an item to help numba infer type
        self.pq = nb.typed.List.empty_list((0.0, 0, (0,0), nb.typed.List([False])))
        self.entry_finder = nb.typed.Dict.empty( (0, 0), (0.0, 0, (0,0), nb.typed.List([False])))
        self.counter = 0

entry_def =  (0.0, 0, nb.types.unicode_type, nb.typed.List([False]))
entry_type = nb.typeof(entry_def)
@jitclass
class PriorityQueueArb(PriorityQueue):
    """Allow for arbitrary keys via string keys"""
    pq: List[entry_type]
    entry_finder: Dict[Tuple[int, int], entry_type]
    counter: int
    entry: entry_type

    def __init__(self):
        # add an item to help numba infer type
        # self.pq = nb.typed.List.empty_list((0.0, 0, nb.types.unicode_type, nb.typed.List([False])))
        # self.entry_finder = nb.typed.Dict.empty( (0, 0), (0.0, 0, nb.types.unicode_type, nb.typed.List([False])))
        self.pq = nb.typed.List.empty_list(entry_def)
        self.entry_finder = nb.typed.Dict.empty(entry_def)
        self.counter = 0


