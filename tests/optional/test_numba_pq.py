import unittest
from heapq import heapify, heappushpop, heappop, heappush
from timeit import timeit
from numba import njit
import numpy as np

from steinerpy.library.search.numba_search_utils import PriorityQueue 

class TestNumbaHeapq(unittest.TestCase):


    def test_numba_pq(self):
        q = PriorityQueue()
        q.put((1,1), 5.0)
        q.put((1,1), 4.0)
        q.put((1,1), 3.0)
        q.put((1,1), 6.0)
        print(q.pq)
        # q.put(1.0)
        # q.put(0.0)
        # for k,v in q.entry_finder.items():
        #     print(k,v.priority)
        print(q.pop())
        print(len(q.entry_finder))

    @njit(cache=True)
    def heapsort(iterable):
        # method to seed the type of the list
        ty = iterable[0]
        h = [ty for _ in range(0)]
        for value in iterable:
            heappush(h, value)
        return [heappop(h) for i in range(len(h))]

    # the following is for informational purposes only
    @njit(cache=True)
    def nlargest(n, it):  # taken from heapq (simplified)
        result = list(it[:n])
        heapify(result)
        for elem in it:
            heappushpop(result, elem)
        #result.sort(reverse=True)
        return result

    @unittest.skip("works, but not needed for now")
    def test_nlargest_optional(self):
        """Running optional heapq test"""
        arr = np.array(10000*[7,3,4,7,2,3,4,6])
        TestNumbaHeapq.nlargest(20, arr) # trigger compilation to cache

        t = timeit(stmt='TestNumbaHeapq.nlargest(20, arr)', number=100, globals={"TestNumbaHeapq": globals().get('TestNumbaHeapq'), 'arr': locals().get('arr')})
        print(t)
        t = timeit(stmt='TestNumbaHeapq.nlargest(20, arr)', number=100, globals={"TestNumbaHeapq": globals().get('TestNumbaHeapq'), 'arr': locals().get('arr')})
        print(t)

    def test_heapsort(self):
        arr = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
        # trigger compilation to cache
        TestNumbaHeapq.heapsort(arr) # trigger compilation to cache
        t = timeit(stmt='TestNumbaHeapq.heapsort(arr)', number=100, globals={"TestNumbaHeapq": globals().get('TestNumbaHeapq'), 'arr': locals().get('arr')})
        print(t)
        t = timeit(stmt='TestNumbaHeapq.heapsort(arr)', number=100, globals={"TestNumbaHeapq": globals().get('TestNumbaHeapq'), 'arr': locals().get('arr')})
        print(t)



if __name__ == "__main__":
    unittest.main(verbosity=2)