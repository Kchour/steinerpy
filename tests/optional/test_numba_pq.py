import unittest
from heapq import heapify, heappushpop, heappop, heappush
# from timeit import timeit
import timeit
from numba import njit
import numba as nb
import numpy as np

from steinerpy.library.search.numba_search_utils import PriorityQueue2D 

class TestNumbaHeapq(unittest.TestCase):

    def test_numba_pq(self):
        q = PriorityQueue2D()
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

        t = timeit.timeit(stmt='TestNumbaHeapq.nlargest(20, arr)', number=100, globals={"TestNumbaHeapq": globals().get('TestNumbaHeapq'), 'arr': locals().get('arr')})
        print(t)
        t = timeit.timeit(stmt='TestNumbaHeapq.nlargest(20, arr)', number=100, globals={"TestNumbaHeapq": globals().get('TestNumbaHeapq'), 'arr': locals().get('arr')})
        print(t)

    def test_heapsort(self):
        arr = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
        # trigger compilation to cache
        TestNumbaHeapq.heapsort(arr) # trigger compilation to cache
        t = timeit.timeit(stmt='TestNumbaHeapq.heapsort(arr)', number=100, globals={"TestNumbaHeapq": globals().get('TestNumbaHeapq'), 'arr': locals().get('arr')})
        print(t)
        t = timeit.timeit(stmt='TestNumbaHeapq.heapsort(arr)', number=100, globals={"TestNumbaHeapq": globals().get('TestNumbaHeapq'), 'arr': locals().get('arr')})
        print(t)

class TestDictVsNumpySpeed(unittest.TestCase):

    @njit
    def numpy_sum():
        arr = np.ones(100000)
        return np.sum(arr)
    
    def reg_np_sum(arr):
        return np.sum(arr)

    @njit
    def dict_sum(my_dict):
        total = 0.0
        for k,v in my_dict.items():
            total += v
        return total

    def reg_dict_sum(my_dict):
        total = 0.0
        for k,v in my_dict.items():
            total += v
        return total

    def test_summing_speed(self):
        arr = np.ones(100000)
        reg_dict = {i:1 for i in range(100000)}
        my_dict = nb.typed.Dict.empty(nb.types.int64, nb.types.int64)
        for i in range(100000):
            my_dict[i] = 1
        # my_dict = {1: 1 for _ in range(100)}

        TestDictVsNumpySpeed.numpy_sum() # trigger compilation to cache
        TestDictVsNumpySpeed.dict_sum(my_dict) # trigger compilation to cache

        t = timeit.repeat(stmt="TestDictVsNumpySpeed.numpy_sum()", number=100, globals={"TestDictVsNumpySpeed":globals().get("TestDictVsNumpySpeed"), "arr": locals().get('arr'), "my_dict": locals().get('my_dict')}, repeat=5)
        print("numba NP: ",t)
        t = timeit.repeat(stmt="TestDictVsNumpySpeed.reg_np_sum(arr)", number=100, globals={"TestDictVsNumpySpeed":globals().get("TestDictVsNumpySpeed"), "arr": locals().get('arr'), "my_dict": locals().get('my_dict')}, repeat=5)
        print("NP: ",t)
        t = timeit.repeat(stmt="TestDictVsNumpySpeed.dict_sum(my_dict)", number=100, globals={"TestDictVsNumpySpeed":globals().get("TestDictVsNumpySpeed"), "arr": locals().get('arr'), "my_dict": locals().get('my_dict')}, repeat=5)
        print("numba Dict: ", t)
        t = timeit.repeat(stmt="TestDictVsNumpySpeed.reg_dict_sum(reg_dict)", number=100, globals={"TestDictVsNumpySpeed":globals().get("TestDictVsNumpySpeed"), "arr": locals().get('arr'), "reg_dict": locals().get('reg_dict')}, repeat=5)
        print("Dict: ", t)




if __name__ == "__main__":
    unittest.main(verbosity=2)