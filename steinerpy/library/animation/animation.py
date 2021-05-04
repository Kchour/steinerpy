""" Animation helper class based on matplotlib.pyplot 

Each plot can have its own figure or draw on the same figure.
However, each class instance will have its own set of data    

References: 
    https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot

"""
import matplotlib.pyplot as plt
import time

class Animate:
    """Animation class to be instantiating, which creates figure and axes objects

    Parameter:
        number (int): Figure number. Using the same number throughout will allow you to plot on the same figure
        xlim (tuple): min-max x dimensions
        ylim (tuple): min-max y dimensions
        gridSize (float, int): Doesn't seem to be used atm
        lineWidth (float, int): Controls line thickness for line plots
        markerType (str): See matplotlib api for more info
        markerSize (int, float): See matplotlib api for more info
        sleep (int, float): controls delay between each frame
        order (int): controls the order in which plots are drawn on top of each (doesn't seem to work well)
        subplot_instance (tuple): (x,y,z), where x,y refers to the grid dimensions, and z is the instance
        fig_size (tuple): Figure size in inches (width, height)

    Todo:
        Need to make this class easier to use!

    """    
    figTrack = {}
    def __init__(self, number=1, xlim=(10,10), ylim=(10,10), gridSize=1, linewidth=5, markerType='o', markerSize=5, sleep=0, order=10, subplot_instance=None, fig_size=None,\
                subplot_title=None, figure_title=None):

        self.xlim = xlim
        self.ylim = ylim
        # Only add subplot if initializing!
        # TODO: Add legend during init phase
        if not plt.fignum_exists(number):
            # Figure doesnt exist, so initialize stuff
        
            # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html
            #         figsize : (float, float), optional, default: None

            # width, height in inches. If not provided, defaults to rcParams["figure.figsize"] = [6.4, 4.8] = [6.4, 4.8].

            #         fig = plt.figure(number, figsize=fig_size)
            # set figure number and size
            fig = plt.figure(number, figsize=fig_size)     

            # Pick specific subplots
            if subplot_instance is None:
                ax1 = fig.add_subplot(1,1,1)
            else:
                ax1 = fig.add_subplot(*subplot_instance)
            
            # Set subplot title
            if subplot_title is not None:
                ax1.title.set_text(subplot_title)

            # set whole figure title
            if figure_title is not None:
                fig.suptitle(figure_title, fontsize=16)    

            # set axis limits    
            if len(xlim) != 0 and len(ylim) != 0:
                ax1.set_xlim(self.xlim[0], self.xlim[1])
                ax1.set_ylim(self.ylim[0], self.ylim[1])
            else:
                ax1.set_autoscalex_on(True)
                ax1.set_autoscaley_on(True)
        else:
            fig = plt.figure(number, figsize=fig_size)
            #ax1 = fig.gca()

            # Pick specific subplots
            if subplot_instance is None:
                ax1 = fig.add_subplot(1,1,1)
            else:
                ax1 = fig.add_subplot(*subplot_instance)

                        # Set subplot title
            if subplot_title is not None:
                ax1.title.set_text(subplot_title)

            # set whole figure title
            if figure_title is not None:
                fig.suptitle(figure_title, fontsize=16)    

        # markerType = 'xc'
        self.markerType = markerType
        self.markerSize = markerSize
        self.linewidth = linewidth
        self.order = order

        self.lines, = ax1.plot([],[],markerType, markersize=markerSize, linewidth=linewidth, zorder=order)
        ax1.set_zorder(order)
        ax1.patch.set_visible(True)    #may not be necessary

        #ax1.grid(True)

        # note that the first draw comes before setting data
        fig.canvas.draw() 
        plt.show(block=False)
        plt.axis('scaled')  #equal is another one

        # Bug when saving animation, keep ax size fixed
        ax1.set_xlim(self.xlim[0], self.xlim[1])
        ax1.set_ylim(self.ylim[0], self.ylim[1])

        # set background color
        ax1.set_facecolor('xkcd:light grey')

        # copy background
        self.axbackground = fig.canvas.copy_from_bbox(ax1.bbox)

        # data holder
        self.xdata = []
        self.ydata = []

        # sleep amount  (secs)
        self.sleep = sleep  

        #Listen to mouse click for pausing
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.pause = False

        # set things for class-wide access
        self.fig = fig
        self.ax1 = ax1   

    def update(self,new_data):
        """ Add new_data, which is a tuple of (x,y). Older data is saved, because we are appending new data
        
        Todo: 
            Find a better way of adding new data. Currently requires us to use numpy like crazy
        """
        if not isinstance(new_data[0],list):
            self.xdata.append(new_data[0])
            self.ydata.append(new_data[1])
        else:
            # + operator also works to extend
            self.xdata.extend(new_data[0])
            self.ydata.extend(new_data[1])
        self.lines.set_xdata(self.xdata)
        self.lines.set_ydata(self.ydata)
        
        #sleep
        time.sleep((self.sleep))

        # restore background
        # self.fig.canvas.restore_region(self.axbackground)

        if self.pause == True:
            plt.waitforbuttonpress()

        # Bug when saving animation, keep ax size fixed
        self.ax1.set_xlim(self.xlim[0], self.xlim[1])
        self.ax1.set_ylim(self.ylim[0], self.ylim[1])

        self.ax1.draw_artist(self.lines)
        # self.fig.canvas.blit(self.ax1.bbox)
        # self.fig.canvas.blit()

        # cache background
        # self.axbackground = self.fig.canvas.copy_from_bbox(self.ax1.bbox)

        self.fig.canvas.update()
        self.fig.canvas.flush_events()

    def onclick(self, event):
        """A mouse click will pause the animation """
        self.pause ^= True
        #fig.canvas.mpl_disconnect(cid)
    
    def update_clean(self, new_data):
        """ Set the current plot to draw the latest data, don't keep old data
        """
        # first remove the previous
        # self.fig.canvas.update()
        # self.fig.canvas.flush_events()

        # Now add the line back in
        # self.lines, = self.ax1.plot([],[], self.markerType, markersize=self.markerSize, linewidth=self.linewidth, zorder=self.order)

        self.lines.set_xdata(new_data[0])
        self.lines.set_ydata(new_data[1])       
        # cache background


        #self.fig.canvas.restore_region(self.axbackground)
        if self.pause == True:
            plt.waitforbuttonpress()

        # Bug when saving animation, keep ax size fixed (or redraw the whole thing)
        self.ax1.set_xlim(self.xlim[0], self.xlim[1])
        self.ax1.set_ylim(self.ylim[0], self.ylim[1])

        self.ax1.draw_artist(self.lines)
        # set background color
        # self.fig.canvas.draw_idle()     # slow but works
        #self.fig.canvas.blit(self.ax1.bbox)
        self.fig.canvas.update()
        self.fig.canvas.flush_events()
        #sleep
        # self.fig.canvas.restore_region(self.axbackground)

        time.sleep((self.sleep))

from matplotlib.animation import FFMpegWriter
class SaveAnimation:
    """ Helper class to help save animations """

    def __init__(self, figNumber, outfile):      
        #self.fig = plt.gcf()    #assuming we are focused on 1 plt
        self.fig = plt.figure(figNumber)
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        self.writer = FFMpegWriter(fps=15, metadata=metadata)
        self.writer.setup(self.fig, outfile, dpi=100)

    def update(self):
        #self.fig = animateObject.fig #may not be necessary here...
        self.writer.grab_frame()

    def save(self):
        self.writer.finish()