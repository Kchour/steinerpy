import numpy as np
import matplotlib.pyplot as plt

# define your x and y arrays to be plotted
t = np.linspace(start=0, stop=2*np.pi, num=100)
y1 = np.cos(t)
y2 = np.sin(t)
y3 = np.tan(t)
plots = [(t,y1), (t,y2), (t,y3)]

# now the real code :) 
curr_pos = 0

def key_event(e):
    global curr_pos

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(plots)

    ax.cla()
    ax.plot(plots[curr_pos][0], plots[curr_pos][1])
    fig.canvas.draw()

fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.add_subplot(111)
ax.plot(t,y1)
plt.show()