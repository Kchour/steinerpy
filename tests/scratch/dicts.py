fire = [ ]
def test():
    return {(i,): {i: 1} for i in range(10)}

for i in range(5):
    fire.append(test())
pass