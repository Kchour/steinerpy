'''test animation class. Can be run as a unittest
'''

# Import things
from steinerpy.library.animation.animation import Animate, SaveAnimation
import numpy as np
T = np.linspace(0,10*np.pi, 100)

# use Animation Class
 
xlim = (0, 10*np.pi)
ylim = (-5, 5)

# 'higher' order will plot on top
# sleep:= pauseing plot for some seconds
plotObject = Animate(number=1, xlim=xlim, ylim=ylim, linewidth=5, markerType='o', markerSize=5, sleep=0.01, order=10 )
plotObjectClean = Animate(number=1, xlim=xlim, ylim=ylim, linewidth=5, markerType='o', markerSize=5, sleep=0.01, order=10 )
plotObject2 = Animate(number=1, xlim=xlim, ylim=ylim, linewidth=5, markerType='o', markerSize=5, sleep=0.01, order=10 )

# using SaveAnimation Class (create a video)
# videoObject = SaveAnimation(figNumber=1, outfile='test_plot.mp4')

for t in T:

    # videoObject.update()      #uses grab_frame method

    # Try to plot without history
    y = np.cos(t)
    plotObjectClean.update_clean((t,y))

    y = np.sin(t) + np.cos(t)
    # Pass in a tuple
    plotObject2.update_clean((t,y))

    y = np.sin(t)
    # Pass in a tuple
    plotObject.update((t,y))

# finish video recording

# videoObject.save()

# keep plot opened
import matplotlib.pyplot as plt
plt.show()
