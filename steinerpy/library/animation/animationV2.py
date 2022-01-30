"""This module provides a helper class for using matplotlib animations """
import matplotlib.pyplot as plt
import time

import steinerpy.config as cfg
    
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

    def _add_artists(self, artist, artist_name, use_line=True):
        if use_line:
            self.artists[artist_name] = {'artist': artist, 'xdata': [], 'ydata': []}    
        else:
            self.artists[artist_name] = {'artist': [artist]}    

    def _on_draw(self, event):
        cv = self.canvas
        fig = cv.figure
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self.background = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()
        cv.blit(fig.bbox)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        # for a in self.artists.values():
        #     fig.draw_artist(a['artist'][0])
        sorted_artist = sorted(self.artists.values(), key=lambda x: x['artist'][0].get_zorder())
        # for a in cls.instances[figure_number].artists.values():
        for a in sorted_artist:
            # Draw artists
            fig.draw_artist(a['artist'][0])

    @classmethod
    def get_artist(cls, artist_name, figure_number=None):
        if figure_number is None:
            figure_number = plt.get_fignums()[-1]
        if artist_name in cls.instances[figure_number].artists:
            return cls.instances[figure_number].artists[artist_name]['artist'][0]

    @classmethod
    def delete(cls, artist_name, figure_number=None):
        """Removes a particular artist from both this class and the axes"""
        if figure_number is None:
            figure_number = plt.get_fignums()[-1]
        if artist_name in cls.instances[figure_number].artists:
            cls.instances[figure_number].artists[artist_name]['artist'][0].remove()
            del cls.instances[figure_number].artists[artist_name]

    # helper method for different user inputs
    @classmethod
    def add_line(cls, *args, **kwargs):
        """Add an line artist to a particular class instance 

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
    def create_new_plot(cls, *args, **kwargs):
        """Return fig, ax from subplots function """
        
        return plt.subplots(*args, **kwargs) 

    @classmethod
    def init_figure(cls, fig, ax, figure_number=None, figure_name="", xlim=None, ylim=None):
        """Allow the user to manually initialize the figure, they must pass in handles

        """
        if figure_number is None:
            figure_number = plt.get_fignums()[-1]

        plt.show(block=False)   
        plt.pause(0.1)

        #resize to prevent marker clipping on the edges
        if xlim != None and ylim != None:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
        cls.prevent_clipping()

        # # try setting equal axis?
        ax.set_aspect("equal")
        
        # Store the background in new class instance
        o = AnimateV2(figure_number, figure_name=figure_name)
        # o.background = fig.canvas.copy_from_bbox(ax.bbox)
        cls.instances[figure_number] = o

    @classmethod
    def _add(cls, artist_name, x, y, *args, figure_number=None, figure_name="", xlim=None, ylim=None, draw_clean=False, linestyle="", alpha=1, **kwargs):
        """Add line2d artist and its data to a particular figure 

        Args:
            artist_name (str): Name of the line2d artist
            x (list of floats, float): The line2d xdata
            y (list of floats, float): The line2d ydata
            args (str): Format arguments for plot 
            xlim (tuple): (xmin, xmax)
            ylim (tuple): (ymin, ymax)
            
        """
        if figure_number is None:
            figure_number = plt.get_fignums()[-1]
        # initialization event.canvas.figure.axes[0].has_been_closed = True
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

            #resize to prevent marker clipping on the edges
            cls.prevent_clipping()

            # Draw the canvas once
            # fig.canvas.draw_idle()
            plt.legend()    # must have already defined this
            plt.show(block=False)    
            plt.pause(0.1)

            # Store the background in new class instance
            o = AnimateV2(figure_number, figure_name=figure_name)
            o.background = fig.canvas.copy_from_bbox(ax.bbox)
            cls.instances[figure_number] = o

        else: 
            # Get figure
            fig = plt.figure(figure_number)
            ax = fig.axes[0]
        
        # Detect when figure is closed. then delete everything basically
        cls.cid_closed_fig = fig.canvas.mpl_connect('close_event', cls.on_shutdown)

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
            if isinstance(x, float) or isinstance(x, int) or "int64" in str(type(x)):
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
        line.set_alpha(alpha)

    @classmethod
    def add_artist_ex(cls, artist, artist_name, figure_number=None):
        """Add any user defined artist """
        # get latest figure if figure_number is not specified
        if figure_number is None:
            figure_number = plt.get_fignums()[-1]

        if artist_name not in cls.instances[figure_number].artists:
            artist.set_animated(True)
            cls.instances[figure_number]._add_artists(artist, artist_name, use_line=False)

    @classmethod
    def update(cls, figure_number=None):
        if figure_number is None:
            figure_number = plt.get_fignums()[-1]

        # Get figure
        fig = plt.figure(figure_number)
        ax = fig.axes[0]

        if cls.instances[figure_number].background is None:
            cls.instances[figure_number]._on_draw(None)
        else:
            # restore background 
            fig.canvas.restore_region(cls.instances[figure_number].background)
            # #Respect z order
            # sorted_artist = sorted(cls.instances[figure_number].artists.values(), key=lambda x: x['artist'][0].get_zorder())
            # # for a in cls.instances[figure_number].artists.values():
            # for a in sorted_artist:
            #     # Draw artists
            #     fig.draw_artist(a['artist'][0])
            
            cls.instances[figure_number]._draw_animated()
            # blit the axes
            fig.canvas.blit(fig.bbox)
        # fig.canvas.update()
        # flush events
        fig.canvas.flush_events()
        # fig.canvas.flush_events()
        # pause if necessary
        if cfg.Animation.animate_delay > 0:
            time.sleep(cfg.Animation.animate_delay)

    
    @classmethod
    def on_shutdown(cls, event):
        # When figure is closed, clear out all figure instances
        cls.instances = {}

    @classmethod
    def close(cls, figure_number=None):
        if figure_number is None:
            figure_number = plt.get_fignums()[-1]

        # delete fig from class
        del cls.instances[figure_number]


        plt.close(figure_number)

    @classmethod
    def prevent_clipping(cls):
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




