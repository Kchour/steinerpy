""" Allows for the use of abstract attributes during class definitions

reference:
    https://stackoverflow.com/questions/23831510/abstract-attribute-not-property/23833055

example:
    < older>
    from better_abc import ABCMeta, abstract_attribute    # see below

    class AbstractFoo(metaclass=ABCMeta):

        @abstract_attribute
        def bar(self):
            pass

    class Foo(AbstractFoo):
        def __init__(self):
            self.bar = 3

    class BadFoo(AbstractFoo):
        def __init__(self):
            pass

    < NEWER>
    from better_abc import ABC, abstract_attribute    # see below

    class AbstractFoo(ABCMeta):

        @abstract_attribute
        def bar(self):
            pass

    class Foo(AbstractFoo):
        def __init__(self):
            self.bar = 3

    class BadFoo(AbstractFoo):
        def __init__(self):
            pass    

"""

from abc import ABCMeta as NativeABCMeta

class DummyAttribute:
    pass

def abstract_attribute(obj=None):
    if obj is None:
        obj = DummyAttribute()
    obj.__is_abstract_attribute__ = True
    return obj


class ABCMeta(NativeABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = NativeABCMeta.__call__(cls, *args, **kwargs)
        missing_abstract_attributes = {
            name
            for name in dir(instance)
            if getattr(getattr(instance, name), '__is_abstract_attribute__', False)
        }
        if missing_abstract_attributes:
            raise TypeError(
                "Can't instantiate abstract class {} with"
                " abstract attributes: {}".format(
                    cls.__name__,
                    ', '.join(missing_abstract_attributes)
                )
            )
        return instance

    
class ABC(metaclass=ABCMeta):
    pass

