from typing import List, Entry, Dict, Tuple

from heapq import heappush, heappop
import numba as nb
from numba import jitclass

@jitclass
class Entry:
    priority: float
    count: int
    item: Tuple[int, int]
    removed: bool

    def __init__(self, p: float, c: int, i: Tuple[int, int]):
        self.priority = p
        self.count = c
        self.item = i
        self.removed = False

    def __lt__(self, other):
        return self.priority < other.priority

@jitclass
class PriorityQueue:
    pq: List[Entry]
    entry_finder: Dict[Tuple[int, int], Entry]
    counter: int

    def __init__(self):
        self.pq = nb.typed.List.empty_list(Entry(0.0, 0, (0, 0)))
        self.entry_finder = nb.typed.Dict.empty((0, 0), Entry(0, 0, (0, 0)))
        self.counter = 0

    def put(self, item: Tuple[int, int], priority: float = 0.0):
        """Add a new item or update the priority of an existing item"""
        if item in self.entry_finder:
            self.remove_item(item)
        self.counter += 1
        entry = Entry(priority, self.counter, item)
        self.entry_finder[item] = entry
        heappush(self.pq, entry)

    def remove_item(self, item: Tuple[int, int]):
        """Mark an existing item as REMOVED.  Raise KeyError if not found."""
        entry = self.entry_finder.pop(item)
        entry.removed = True

    def pop(self):
        """Remove and return the lowest priority item. Raise KeyError if empty."""
        while self.pq:
            priority, count, item = heappop(self.pq)
            entry = heappop(self.pq)
            if not entry.removed:
                del self.entry_finder[entry.item]
                return item
        raise KeyError("pop from an empty priority queue")
