from timeit import default_timer as timer
import numpy as np


# larger matrix
test_3d1 = np.zeros((100, 100, 100))
obs = np.array([(i,i,i) for i in range(100)])

# slower method
# test using loop
# for o in obs:
#     test_3d1[o] = 1
# faster
test_3d1[obs[:,0], obs[:,1], obs[:,2]] = 1

# # slow loop method
# t1 = timer()
# samples = set()
# while len(samples) < 500e3:
#     x,y,z = np.random.randint((0, 0, 0), (100, 100, 100))
#     if test_3d1[x,y,z] < 1:
#         samples.add((x,y,z))

# # convert to list
# samples = list(samples)

# print("Slow method: ", timer()-t1, "len: ", len(samples))
# del samples

# slightly faster?
t1 = timer()
samples = set()
desired_size = int(500e3)
size = int(500e3)
while len(samples) < desired_size:
    gen = np.random.randint((0, 0, 0), (100, 100,  100), size=(size,3))
    get = test_3d1[gen[:,0], gen[:,1], gen[:,2]] < 1
    items = gen[get]
    for i in items:
        if len(samples)<desired_size:
            samples.add(tuple(i))
        else:
            break
    size = desired_size - len(samples)

# convert to list
samples = list(samples)
print("Alternate: ", timer()-t1, "len: ",len(samples))


t1 = timer()
test = {}
for i in range(10000):
    for j in range(1000):
        test[(i,j)] = i*j
print("dict creationg: ", timer()-t1)

t1 = timer()
test = np.full((10000, 1000), np.inf)
for i in range(10000):
    for j in range(1000):
        test[i,j] = i*j

print("np creating and setting", timer()-t1)

# # try random sampling
# # ind = np.where(test_3d1<1)
# ind = test_3d1<1
# sample_ind = np.random.choice(np.arange(len(ind[0])), 5, replace=False)
# x = ind[0][sample_ind] 
# y = ind[1][sample_ind]
# z = ind[2][sample_ind]

# s = np.vstack((x,y,z)).T
# # list of tuples
# lot = [tuple(x) for x in s]


pass

