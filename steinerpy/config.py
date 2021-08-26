"""Configure global flags here. Any flag changes will be captured
    across all modules. Users must import the entire module via
    import steinerpy.library.config

    Note: python modules are considered singletons

"""
import logging, logging.config

class _ColoredFormatter(logging.Formatter):
    """Customized custom formatter class to allow colored outputs to the console!
        No need to import this at all!
    """
    # logging colors
    DEBUG = "\033[92m"  # LIGHT GREEN
    INFO = "\033[94m"   # LIGHT BLUE
    # DEBUG = '\033[32m'    # GREEN
    # INFO = "\033[97m"     # BLUE
    WARNING = "\033[33m"    # YELLOW
    ERROR = "\033[31m"   # RED
    CRITICAL = '\033[91m'      # High-intensity RED

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'        # RESET
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # output format
    format = "%(asctime)-15s : %(levelname)-8s : %(name)-15s : %(message)s"

    # new format map
    COLOR_FMT = {
        'WARNING': "".join((WARNING, format, ENDC)),
        'INFO': "".join((INFO, format, ENDC)),
        'DEBUG': "".join((DEBUG, format, ENDC)),
        'CRITICAL': "".join((CRITICAL, format, ENDC)),
        'ERROR': "".join((ERROR, format, ENDC)),
    }

    def format(self, record):
        """override format method"""
        # get name and level of recording
        levelname= record.levelname
        levelno = record.levelno

        # get new format from map
        new_fmt = _ColoredFormatter.COLOR_FMT[levelname]

        # create a new formatter object
        formatter = logging.Formatter(new_fmt)

        # return it wrapped around record!
        return formatter.format(record)


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
    # Run additional functions in framework.py
    DEBUG_MODE = False

    # Sound alert
    sound_alert = False

    # Profile framework module (print runtime of code)
    profile_frame = False

    # This can be configured by the user!
    log_conf = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'default': {
                'format': '%(asctime)-15s %(levelname)-8s %(name)-8s %(message)s'
            },
            'colored':{
                # https://docs.python.org/3/library/logging.config.html#dictionary-schema-details
                # alternatively, steinerpy.library.config.ColoredFormatter
                '()':  _ColoredFormatter
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'colored',
                'level': "WARNING"
            },
            'file_handler': {
                'class': 'logging.FileHandler',
                'formatter': 'default',
                'level': "WARNING",
                'filename': '/tmp/my_steinerpy_logfile.log'
            },
        },
        'loggers': {
            # You can create a name of logger you desire
            '': {# root logger
                'handlers': ['file_handler', 'console'],
                'level': 'DEBUG'
            },

        },
    }

class Animation:
    # visualize plot
    visualize = False
    save_movie = False  # Not implemented
    animate_delay = 0.0

class Algorithm:
    # Algo runtime settings
    sstar_heuristic_type = "diagonal_nonuniform" # grid-based heuristics (manhattan, euclidean, diagonal_uniform, diagonal_nonuniform, preprocess, custom)
                                              # zero otherwise
    hFactor = 1.0    # Scalng factor for heuristics (located in common.grid_based_heuristics)
    graph_domain = "grid" # grid, generic
    
    ########################################################
    # The following operations change a component's frontier costs
    # which theoretically helps reduce node expansions.
    # In practice, this increases overhead. 
    # Works well when heuristics are really good. 
    # Ultimately performance depends on the problem instance with
    # factors like how close terminals are, obstacle positions, etc..
    #########################################################

    # Change the 2 components' frontier costs corresponding to a merge
    reprioritize_after_merge = True     
    # Change the 2 components' frontier costs corresponding to a shortest path
    reprioritize_after_sp = False       
    

########################################################
# DEFAULT SETTINGS
########################################################


def reload_log_conf():
    """Call this to explicitly reconfigure the logging module,
        especially if log_conf is changed by the user!
    """
    logging.config.dictConfig(Misc.log_conf)
reload_log_conf()

