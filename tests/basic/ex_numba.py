from numba.experimental import jitclass
from numba import jit, types, typed #int64
from timeit import default_timer as timer
import unittest
import numpy as np

# Intesive for loop
def slow_f(x, y):
    sum = 0
    for i in range(x):
        sum+=i
        for j in range(y):
            sum+=j
    return sum

# Optimized loop
@jit(nopython=True)
def jit_f(x, y):
    sum = 0
    for i in range(x):
        sum+=i
        for j in range(y):
            sum+=j
    return sum

# Trying numba jitclass
spec = [('x', types.int64),
        ('y', types.int64),
        ('sum', types.int64)]
@jitclass(spec)
class JITClassTest(object):
    def __init__(self,x, y):
        self.x = x
        self.y = y
        self.sum = 0
    
    def intense_loop(self):
        for i in range(self.x):
            self.sum+=i
            for j in range(self.y):
                self.sum+=j
        return self.sum

# Trying numba jitclass class variables
class JITClassVariableTest(object):
    x = 10000
    y = 20000
    sum = 0
    
    # must define a wrapper for actual function
    @classmethod
    def intense_loop(cls):
        return cls._intense_loop(cls.x,cls.y,cls.sum)

    @staticmethod
    @jit(nopython=True)
    def _intense_loop(x,y,sum):
        for i in range(x):
            sum+=i
            for j in range(y):
                sum+=j
        return sum

# Example JIT with dict
@jit(nopython=True)
def example_jit_dict():
    # value_type = np.int64 also works here
    dict_ = typed.Dict.empty(
        key_type = types.unicode_type,
        value_type = types.int64
    )
    alphabet = ["a", "b", "c", "d"]
    numbers = range(4)

    for k, v in zip(alphabet, numbers):
        dict_[k] = v
    # dict_ = {k:v for k,v in zip(alphabet, numbers)}   # Does not work with numba

    return dict_

# JIT modifying dict


@jit(nopython=True)
def modify_jit_dict(d_param1):
    # create a new dict to return data
    # dict_ = typed.Dict.empty(
    #     key_type = types.unicode_type,
    #     value_type = types.int64
    # )

    # _return_dict = {
    # dict_['e'] = 5
    # return dict_

    d_param1['e'] = 5
    return d_param1


# Testing JIT with custom dict. REALLY SLOW
@jit(nopython=True)
def JITwithDictSlow(d_param1):    

    for i in range(d_param1['x']):
        d_param1["sum"] += i
        for j in range(d_param1['y']):
            d_param1["sum"] += j

    return d_param1 

# Testing JIT with custom dict. Faster
@jit(nopython=True)
def JITwithDictFast(d_param1):    
    x = d_param1['x']
    y = d_param1['y']
    sum_ = d_param1['sum']
    for i in range(x):
        sum_ += i
        for j in range(y):
            sum_ += j

    d_param1['sum'] = sum_
    return d_param1 


# # Using dictionaries 
# class JITClassWithDictJIT:
#     data = {'x': 10000, 'y': 20000, 'sum':0}

#     # dict_type = nb.deferred_type()
#     # dict_type.define(nb.typeof(nb.typed.Dict.empty(key_type=nb.types.unicode_type, value_type=nb.types.int64)))

#     @classmethod
#     def intense_loop(cls):
#         return cls._intense_loop(cls.data)

#     @staticmethod
#     @jit(nopython=True)
#     def _intense_loop(data):
#         data = typed.Dict.empty(key_type=types.unicode_type, value_type=types.int64)
#         for i in range(data['x']):
#             data['sum']+= i
#             for j in range(data['y']):
#                 data['sum']+= j
#         return data    

####### Main tests ########
@unittest.skip("SKIPPING TestJITFeatures (NUMBA)")
class TestJITFeatures(unittest.TestCase):

    ### Slowest
    # def test_slow_intensive_forloops(self):
    #     start = timer()
    #     val = slow_f(10000,20000)
    #     end = timer()
    #     print("non-jit time taken (s):", end - start)
    #     self.assertEqual(val, 1999949995000)

    ### Fastest
    # def test_jit_intensive_forloops(self):
    #     # Starts to matter when computation is very intensive
    #     start = timer()
    #     val = jit_f(10000,20000)
    #     end = timer()
    #     print("jit time taken (s):", end - start)
    #     self.assertEqual(val, 1999949995000)

    ### Fast
    # def test_class_jit(self):
    #     x = 10000
    #     y = 20000
    #     jo = JITClassTest(x, y)
    #     start = timer()
    #     val = jo.intense_loop()
    #     end = timer()
    #     print("jit_class intense_loop time taken (s):", end - start)
    #     print("val", val)
    #     self.assertEqual(val, 1999949995000)

    ### Fast
    def test_class_variables_jit(self):
        start = timer()
        val = JITClassVariableTest.intense_loop()
        end = timer()
        print("JIT class variables intense_loop time taken (s):", end-start)
        self.assertLessEqual(end-start, 2)
        self.assertEqual(val, 1999949995000)

    def test_example_jit_dict(self):
        start = timer()
        val = example_jit_dict()
        end = timer()
        print("test example jit dict, time taken (s):", end-start)
        print(val)

    def test_modifying_jit_dict(self):
        d_param1 = typed.Dict.empty(
            key_type = types.unicode_type,
            value_type = types.int64
        )
        alphabet = ["a", "b", "c", "d"]
        numbers = range(4)
        # List comprehension wont work!
        for k, v in zip(alphabet, numbers):
            d_param1[k] = v

        start = timer()
        val = modify_jit_dict(d_param1)
        end = timer()
        print("test modifying jit dict, time taken (s):", end-start)
        print(val)

    ### MODIFYING DICTS IS REALLY SLOW
    def test_JIT_with_dict_slow(self):
        d_param1 = typed.Dict.empty(
            key_type = types.unicode_type,
            value_type = types.int64
        )
        # MUST USE any empty dict() or {} 
        keys = ["x", "y", "sum"]
        vals = [10000, 20000, 0]
        for k,v in zip(keys, vals):
            d_param1[k] = v

        start = timer()
        val = JITwithDictSlow(d_param1)
        end = timer()
        print("JIT with dict, time taken (s):", end-start)
        self.assertEqual(val['sum'], 1999949995000)
        self.assertGreaterEqual(end-start, 2)

    ### SO DON'T MODIFY/ACCESS DICTS DURING LOOP!
    def test_JIT_with_dict_fast(self):
        d_param1 = typed.Dict.empty(
            key_type = types.unicode_type,
            value_type = types.int64
        )
        # MUST USE any empty dict() or {} 
        keys = ["x", "y", "sum"]
        vals = [10000, 20000, 0]
        for k,v in zip(keys, vals):
            d_param1[k] = v

        start = timer()
        val = JITwithDictFast(d_param1)
        end = timer()
        print("JIT with dict, time taken (s):", end-start)
        self.assertEqual(val['sum'], 1999949995000)
        self.assertLessEqual(end-start, 2)

    ### NOT WORKING YET
    # def test_class_variables_with_dict_jit(self):
    #     start = timer()
    #     val = JITClassWithDictJIT.intense_loop()
    #     end = timer()
    #     print("JIT class variable with dict, time taken (s):", end-start)
    #     self.assertLessEqual(end-start, 2)
    #     self.assertEqual(val['sum'], 1999949995000)

# class JIT:

#     @jit(float64(float64, float64), nopython=True)
#     def jit_f(self,x, y):
#         self.sum = 0
#         for i in range(x):
#             self.sum+=i
#             for j in range(y):
#                 self.sum+=j
#         return self.sum

# def jit_f(self,x, y):
#         self.sum = 0
#         for i in range(x):
#             self.sum+=i
#             for j in range(y):
#                 self.sum+=j
#         return self.sum



if __name__=="__main__":
    unittest.main()

