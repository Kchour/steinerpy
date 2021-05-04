""" If you change the config variable before importing, then
    you can set it. However after importing, it cannot be changed
"""

import unittest
import numpy as np

from steinerpy.library.animation.animationV2 import AnimateV2

class TestConfig(unittest.TestCase):
    
    def test_update_and_cleandraw_xy(self):
        print("test_update_and_cleandraw_xy")
        xo = np.linspace(-15,15, 150)
        yo = np.cos(xo)
        zo = np.cos(yo)

        for x,y,z in zip(xo, yo, zo): 
            # Add things
            AnimateV2.add_line("cos", x, y, markersize=5, marker='o', zorder=10) #on top
            AnimateV2.add_line("sin", x, z, draw_clean=True, markersize=10, marker='o')
            
            # Update figure
            AnimateV2.update()

        # Close figure when done
        AnimateV2.close()

    def test_delete_xy(self):
        print("test_delete_xy")
        xo = np.linspace(-15,15, 150)
        yo = np.cos(xo)
        zo = np.sin(xo)
        to = np.sin(xo) + np.cos(xo)

        for a,b in enumerate(zip(xo,yo,zo,to)): 
            x,y,z,t = b
            # Add things
            AnimateV2.add_line("cos", x, y, markersize=5, marker='o', zorder=10) #on top
            AnimateV2.add_line("sin+cos", x, t, draw_clean=True, markersize=10, marker='o')
            
            # if a > half delete sin!
            if a >= 125:
                AnimateV2.delete("sin")
            else:
                AnimateV2.add_line("sin", x, z, draw_clean=True, markersize=10, marker='o')

            # Update figure
            AnimateV2.update()
        
        # Close figure when done
        AnimateV2.close()

    def test_variable_number_of_inputs(self):
        print("test_variable_number_of_inputs")
        xo = np.linspace(-15,15,150)
        yo = np.cos(xo)
        data = np.array([xo,yo]).T.tolist()

        for d in data:
            # Add artists
            AnimateV2.add_line('cos', d, markersize=5, marker='o')

            # Update canvas drawings      
            AnimateV2.update()

        AnimateV2.close()


    def test_multiple_xy_input(self):
        print("test_multiple_xy_input")
        xo = np.linspace(-15,15,150)
        yo = np.cos(xo)
        data = np.array([xo,yo]).T

        skip = 5
        for ndx in range(0, len(data), skip):
            if ndx + skip >= len(data):
                break
            # Add artists
            AnimateV2.add_line('cos', data[ndx:ndx+skip,0].tolist(), data[ndx:ndx+skip,1].tolist(), markersize=5, marker='o')

            # Update canvas drawings      
            AnimateV2.update()

        AnimateV2.close()

    def test_multiple_xy_input_with_args(self):
        print("test_multiple_xy_input")
        xo = np.linspace(-15,15,150)
        yo = np.cos(xo)
        data = np.array([xo,yo]).T

        skip = 5
        for ndx in range(0, len(data), skip):
            if ndx + skip >= len(data):
                break
            # Add artists
            AnimateV2.add_line('cos', data[ndx:ndx+skip,0].tolist(), data[ndx:ndx+skip,1].tolist(), 'ro', markersize=5)

            # Update canvas drawings      
            AnimateV2.update()

        AnimateV2.close()

    def test_marker_color_and_size(self):
        print("test_marker_color_and_size")
        xo = np.linspace(-15,15, 150)
        yo = np.cos(xo)
        zo = np.cos(yo)

        for x,y,z in zip(xo, yo, zo): 
            # Add things
            AnimateV2.add_line("cos", x, y, markersize=15, marker='o', color='b', zorder=10) #on top
            AnimateV2.add_line("sin", x, z, draw_clean=True, markersize=10, marker='o', color='r')
            
            # Update figure
            AnimateV2.update()

        # Close figure when done
        AnimateV2.close()

    def test_marker_color_and_size_with_args(self):
        print("test_marker_color_and_size_with_args")
        xo = np.linspace(-15,15, 150)
        yo = np.cos(xo)
        zo = np.cos(yo)

        for x,y,z in zip(xo, yo, zo): 
            # Add things
            AnimateV2.add_line("cos", x, y, 'bo', markersize=15, zorder=10) #on top
            AnimateV2.add_line("sin", x, z, 'o', draw_clean=True, markersize=10)
            
            # Update figure
            AnimateV2.update()

        # Close figure when done
        AnimateV2.close()

    def test_marker_only_plot_no_lines(self):
        print("test_marker_only_plot_no_lines")
        xo = np.linspace(-15,15, 5)
        yo = np.cos(xo)
        zo = np.cos(yo)

        for x,y,z in zip(xo, yo, zo): 
            # Add things
            AnimateV2.add_line("cos", x, y, 'bo', markersize=15, zorder=10) #on top
            AnimateV2.add_line("sin", x, z, 'o', draw_clean=True, markersize=10)
            
            # Update figure
            AnimateV2.update()

        # Close figure when done
        AnimateV2.close()

    def test_compact_data_with_args(self):
        print("test_compact_data_with_args")
        xo = np.linspace(-15,15,250)
        yo = np.cos(xo)
        data = np.array([xo,yo])

        # Add artists
        AnimateV2.add_line('cos', data.tolist(), 'ro', markersize=5)

        # Update canvas drawings      
        AnimateV2.update()

        # Close figure when done
        AnimateV2.close()

if __name__ == "__main__":
    unittest.main()