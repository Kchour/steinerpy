from abc import abstractmethod
from steinerpy.library.misc.abc_utils import abstract_attribute, ABC as newABC

import unittest

# test newABC
class AbstractFoo(newABC):
    bar = abstract_attribute()

class Foo(AbstractFoo):
    def __init__(self):
        self.bar = 3
    
    def return_true(self):
        return True

class BadFoo(AbstractFoo):
    def __init__(self):
        pass

# test newABC with @abstactmethod    
class AbstractFoo2(newABC):
    # @abstract_attribute()
    # def bar(self):
    #     pass
    bar = abstract_attribute

    @abstractmethod
    def cool(self):
        pass 

class Foo2(AbstractFoo2):
    def __init__(self):
        self.bar = 3
        
    def cool(self):
        print('well played')
    
    def return_true(self):
        return True

class BadFoo2(AbstractFoo2):
    def __init__(self):
        self.bar = 3
    
class TestAbstractProperty(unittest.TestCase):

    def test_good_foo(self):
        try: 
            test = Foo()
            self.assertTrue(True)
        except:
            pass

    def test_bad_foo(self):
        with self.assertRaises(TypeError):
            BadFoo()

    def test_good_foo2_abstractmethod(self):
        try: 
            test = Foo2()
            self.assertTrue(True)
        except:
            pass

    def test_bad_foo2_abstractmethod(self):
        with self.assertRaises(TypeError):
            BadFoo2()

if __name__ == "__main__":
    unittest.main()