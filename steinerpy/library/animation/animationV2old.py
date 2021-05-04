"""This module provides a helper class for using matplotlib animations """
import matplotlib.pyplot as plt
import time

from steinerpy.library.config import Config as cfg
    
class AnimateV2:
    """Helper class for matplotlib.pyplot animations
        Class instances are used to keep track of different figures and their
        respective artists
    
    Args:
        figure_number (int): Figure number to put our canvas on
        figure_name (str): NOT IMPLEMENTED YET

    Attributes:
        figure_number
        figure_name (str): NOT IMPLEMENTED YET
        background (bbox): The canvas background
        artists (dict): Container of different artists
        canvas (???): The current figure's canvas
        cid (???): Callback object for "draw_event"

    Todo:
        * Add support for subplots
        * Test usage multiple figures

    """
    # keep track of figure instances
    instances = {}

    def __init__(self, figure_number, figure_name=""):
        self.figure_number = figure_number
        self.figure_name = figure_name
        self.background = None
        self.artists = {}

        # grab the background on every draw            
        fig = plt.figure(figure_number)     
        self.ax = fig.axes[0]
        self.canvas = fig.canvas
        self.cid = self.canvas.mpl_connect("draw_event", self._on_draw)

    def _add_artists(self, artist, artist_name):
        self.artists[artist_name] = {'artist': artist, 'xdata': [], 'ydata': []}    

    def _on_draw(self, event):
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self.background = cv.copy_from_bbox(cv.figure.bbox)

    @classmethod
    def delete(cls, artist_name, figure_number=1):
        """Removes a particular artist from both this class and the axes"""
        if artist_name in cls.instances[figure_number].artists:
            cls.instances[figure_number].artists[artist_name]['artist'][0].remove()
            del cls.instances[figure_number].artists[artist_name]

    # helper method for different user inputs
    @classmethod
    def add(cls, *args, **kwargs):
        """Add an artist to a particular class instance 

        Examples:
            Let x, y be single float values or a list of floats
            >>> AnimateV2.add("cos", x, y, 'bo', markersize=15, zorder=10) #on top

            Let d be a 2D list of floats, i.e. [[x1, x2, ...],[y1, y2, ...]]
            >>> AnimateV2.add('cos', d, markersize=5, marker='o')
            >>> AnimateV2.add('cos', d, 'ro', markersize=5)

        """

        if isinstance(args[-1], str):
            #using fmt arguments
            
            # compact?
            if len(args[0:-1]) == 2:
                artist_name, data = args[0], args[1]
                x,y = data[0], data[1]

                cls._add(artist_name, x,y, args[-1], **kwargs)
            else:
                #not compact?
                cls._add(*args, **kwargs)
        else:
            #not using fmt args
            
            # compact?
            if len(args)==2:
                artist_name, data = args
                x,y = data[0], data[1]
                cls._add(artist_name, x, y, **kwargs)
            else:
                #not compact
                artist_name, x, y = args
                cls._add(artist_name, x, y, **kwargs)

    @classmethod
    def _add(cls, artist_name, x, y, *args, figure_number=1, figure_name="", xlim=None, ylim=None, draw_clean=False, linestyle="", **kwargs):
        """Add line2d artist and its data to a particular figure 

        Args:
            artist_name (str): Name of the line2d artist
            x (list of floats, float): The line2d xdata
            y (list of floats, float): The line2d ydata
            args (str): Format arguments for plot 
            xlim (tuple): (xmin, xmax)
            ylim (tuple): (ymin, ymax)
            
        """
        # initialization    event.canvas.figure.axes[0].has_been_closed = True
        if not plt.fignum_exists(figure_number):
            # Get figure
            fig = plt.figure(figure_number)
            # Add axes
            ax = fig.add_subplot(1,1,1)
            # set limits
            if xlim is None or ylim is None:
                xlim = (-15,15)
                ylim = (-15,15)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])


            ### prevent edge clipping of markers. Save ticks, but change limits
            xticks, xticklabels = plt.xticks()
            yticks, yticklabels = plt.yticks()

            # shaft half a step to the left
            xmin = (3*xticks[0] - xticks[1])/2.
            # shaft half a step to the right
            xmax = (3*xticks[-1] - xticks[-2])/2.

            # shaft half a step below
            ymin = (3*yticks[0] - yticks[1])/2.
            # shaft half a step above
            ymax = (3*yticks[-1] - yticks[-2])/2.

            plt.xlim(xmin, xmax)
            plt.xticks(xticks)

            plt.ylim(ymin, ymax)
            plt.yticks(yticks)

            ######################################

            # Draw the canvas once
            fig.canvas.draw_idle()
            plt.pause(0.75)

            plt.show(block=False)
            # Store the background in new class instance
            o = AnimateV2(figure_number=1, figure_name=figure_name)
            o.background = fig.canvas.copy_from_bbox(ax.bbox)
            cls.instances[figure_number] = o

            # Detect when figure is closed. then delete everything basically
            cls.cid_closed_fig = fig.canvas.mpl_connect('close_event', cls.on_shutdown)
        else: 
            # Get figure
            fig = plt.figure(figure_number)
            ax = fig.axes[0]
        
        # Add artist if not yet
        if artist_name not in cls.instances[figure_number].artists:
            if not args:
                if kwargs:
                    cls.instances[figure_number]._add_artists(ax.plot(x, y, linestyle=linestyle,**kwargs), artist_name)
                else:
                    cls.instances[figure_number]._add_artists(ax.plot(x, y, linestyle=linestyle), artist_name)
            else:
                if kwargs:
                    cls.instances[figure_number]._add_artists(ax.plot(x, y, args[0], linestyle=linestyle, **kwargs), artist_name)
                else:
                    cls.instances[figure_number]._add_artists(ax.plot(x, y, args[0], linestyle=linestyle), artist_name)

        # store data
        if not draw_clean:
            if isinstance(x, float) or isinstance(x, int):
                cls.instances[figure_number].artists[artist_name]['xdata'].append(x)
                cls.instances[figure_number].artists[artist_name]['ydata'].append(y)
            else:
                cls.instances[figure_number].artists[artist_name]['xdata'].extend(x)
                cls.instances[figure_number].artists[artist_name]['ydata'].extend(y)
        else:
            cls.instances[figure_number].artists[artist_name]['xdata'] = x
            cls.instances[figure_number].artists[artist_name]['ydata'] = y

        line = cls.instances[figure_number].artists[artist_name]['artist'][0]
        # Set line2d data
        line.set_xdata(cls.instances[figure_number].artists[artist_name]['xdata'])
        line.set_ydata(cls.instances[figure_number].artists[artist_name]['ydata'])
        
    @classmethod
    def update(cls, figure_number=1):
        # Get figure
        fig = plt.figure(figure_number)
        ax = fig.axes[0]
        # restore background 
        fig.canvas.restore_region(cls.instances[figure_number].background)
        #Respect z order
        sorted_artist = sorted(cls.instances[figure_number].artists.values(), key=lambda x: x['artist'][0].get_zorder())
        # for a in cls.instances[figure_number].artists.values():
        for a in sorted_artist:
            # Draw artists
            ax.draw_artist(a['artist'][0])
        # blit the axes
        fig.canvas.blit(ax.bbox)
        # fig.canvas.update()
        # flush events
        fig.canvas.flush_events()
        # pause if necessary
        plt.pause(0.5)
        if cfg.animate_delay > 0:
            time.sleep(cfg.animate_delay)

    @classmethod
    def on_shutdown(cls, event):
        # When figure is closed, clear out all figure instances
        cls.instances = {}

    @classmethod
    def close(cls):
        plt.close()




