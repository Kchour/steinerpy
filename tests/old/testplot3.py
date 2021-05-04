import numpy as np
import matplotlib.pyplot as plt

# Generate random data
data = np.random.rand(100)

# Plot in different subplots
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(data)

ax2.plot(data)

ax1.plot(data+1)

plt.show()