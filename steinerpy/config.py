"""Configure global flags here. Any flag changes will be captured
    across all modules. Users must import the entire module via
    import steinerpy.library.config

    Note: python modules are considered singletons

"""

# Package location for paths
import steinerpy
import os
pkg_dir = os.path.dirname(steinerpy.__file__)
# file_location =  os.path.dirname(steinerpy.__file__)+"../logs/"
data_dir = pkg_dir + "/../data"
results_dir = pkg_dir + "/../results"
logs_dir = pkg_dir + "/../logs"

testVarOutside = "HungryHippos"


class Misc:
    """Static class for runtime configuration flags"""
    # test variables
    testVar = "HungryHippos"

    # Logger level
    """ WARNING: CAN IMPACT PERFORMANCE
        file_level: file log level (not implemented yet)
        console_level: Filter out console logs according to level
            (CRITICAL, ERROR, WARNING, INFO, DEBUG) 
    """
    file_level = "DEBUG"     
    console_level = "WARNING"

    # Sound alert
    sound_alert = False

    # Profile framework module (print runtime of code)
    profile_frame = False

class Animation:
    # visualize plot
    visualize = False
    save_movie = False  # Not implemented
    animate_delay = 0.0

class Algorithm:
    # Algo runtime settings
    sstar_heuristic_type = "diagonal_uniform" #grid-based heuristics (manhattan, euclidean, diagonal_uniform, diagonal_nonuniform, preprocess, custom)
    hFactor = 1.0    # Scalng factor for heuristics (located in common.grid_based_heuristics)
    