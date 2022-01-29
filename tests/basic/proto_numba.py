from pickletools import float8
from this import d
from typing import final
from unittest import result
from numba import njit,jit, typed, types
import numba as nb

# # Example JIT with dict
# @jit(nopython=True)
# def example_jit_dict():
#     # value_type = np.int64 also works here
#     dict_ = typed.Dict.empty(
#         key_type = types.unicode_type,
#         value_type = types.int64
#     )
#     alphabet = ["a", "b", "c", "d"]
#     numbers = range(4)

#     for k, v in zip(alphabet, numbers):
#         dict_[k] = v
#     # dict_ = {k:v for k,v in zip(alphabet, numbers)}   # Does not work with numba

#     return dict_

# print(example_jit_dict())


# 2-tuple keys and float64 values
def return_empty_dict():
    return typed.Dict.empty(
        key_type=types.UniTuple(types.int64, 3),
        value_type=types.float64,
    )


dict_param1 = typed.Dict.empty(
    key_type=types.UniTuple(types.int64, 3),
    value_type=types.float64,
)

dict_param2 = typed.Dict.empty(
    key_type=types.UniTuple(types.int64, 3),
    value_type=types.float64,
)

nested_dict_param3 = typed.Dict.empty(
    key_type = types.UniTuple(types.int64, 3),
    value_type=nb.typeof(dict_param1)
)

# type-expressions are currently not supportd inside jit functions?
_float64 = types.float64
# _unituple = types.UniTuple(types.int64, 3)

def dim(n):
    """return a uni tuple of int64 of dim n"""
    return types.UniTuple(types.int64, n)

_unituple = dim(3)

@njit
def add_values(d_param1, d_param2):
    # Make a result dictionary to store results
    # Dict with keys as string and values of type float array
    result_dict = typed.Dict.empty(
        key_type=_unituple,
        value_type=_float64,
    )

    for key in d_param1.keys():
      result_dict[key] = d_param1[key] + d_param2[key]

    return result_dict

dict_param1[(1,1,1)] = 10.05
dict_param1[(1,1,2)] = 5.05

dict_param2[(1,1,1)] = 10.05
dict_param2[(1,1,2)] = 5.05

# final_dict = add_values(dict_param1, dict_param2)

# print(final_dict)

dict_type = nb.typeof(dict_param1)

class TestNumba:

    a = 0

    # dict_param3 = typed.Dict.empty(
    #     key_type=types.UniTuple(types.int64, 3),
    #     value_type=types.float64,
    # )
    dict_param3 = {}
    dict_param3[(1,1,1)] = 1000
    dict_param3[(1,1,2)] = 25

    @njit
    def _add_values(d_param1, d_param2, a, key):
        # Make a result dictionary to store results
        # Dict with keys as string and values of type float array


        result_dict = typed.Dict.empty(
            key_type=_unituple,
            value_type=_float64,
        )

        # for key in d_param1.keys():
        #     result_dict[key] = d_param1[key] + d_param2[key] + a
        if key in d_param1:
            result_dict[key] = d_param1[key] + d_param2[key] + a


        return result_dict

    @njit
    def _nested_copy(d_param1):
        # result_dict = n
        d1 = typed.Dict.empty(
            key_type=_unituple,
            value_type=_float64,
        )

        d2 = typed.Dict.empty(
            key_type=_unituple,
            value_type=dict_type, # base the d2 instance values of the type of d1
        )

        for key,values in d_param1.items():
            for key2, v in values.items():
                d1[key2] = v
            d2[key] = d1

        return d2

    def add_values(self, d1, d2, k):
        dict_param3 = typed.Dict.empty(
            key_type=types.UniTuple(types.int64, 3),
            value_type=types.float64,
        )

        for k,v in TestNumba.dict_param3.items():
            dict_param3[k] = v

        # return TestNumba._add_values(d1, d2, TestNumba.a, k)
        # return TestNumba._add_values(d1, TestNumba.dict_param3, TestNumba.a, k)
        return TestNumba._add_values(d1, dict_param3, TestNumba.a, k)

obj = TestNumba()
TestNumba.a = 50
# final_dict = obj.add_values(dict_param1, dict_param2, (1,1,1))
# print(final_dict)

nested_dict_param3[(1,1,1)] = dict_param1
final_dict = TestNumba._nested_copy(nested_dict_param3)
print(final_dict)
