
# Create a decorator
def my_decorations(function):
    def wrapper(*args, **kwargs):
        if args[0].a > 1:
            print("Stop because self.a is greater than 1")
        else:
            return function(*args, **kwargs)
    return wrapper

class TestDecorator:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @my_decorations
    def print_things(self,a,b):
        print(self.a,self.b)
        print(a,b)
        return 1000

if __name__ == "__main__":
    cl = TestDecorator(1,2)
    test = cl.print_things(3,5)
    print("")
