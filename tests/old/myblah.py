import numpy as np
cnt = 0
while cnt < 100:

    blah = np.random.rand(np.random.randint(1,10),np.random.randint(1,10))
    max_min = np.max(np.min(blah, axis=1))  # min over each column, then max
    min_max = np.min(np.max(blah, axis=0))  # max over each row first, then minmize
    # clearly max_min < min_max for any matrix
    print(max_min, min_max, max_min<=min_max)
    cnt += 1