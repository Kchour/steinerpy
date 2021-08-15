"""DEPRECATED: This module provides an interface to the built-in logging module"""

import logging 
import steinerpy.config as cfg
import os
import sys

class Ccolors:
    """ANSI escape sequences for printing colorful texts to console

    References:
        https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
        https://github.com/whitedevops/colors/blob/master/colors.go

    Example:
        print(bcolors.WARNING + "Warning: No active frommets remain. Continue?" + bcolors.ENDC)
        
        or
        
        print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")
    
    Info:
        LIGHTGREEN: Debug
        WHITE: Info
        YELLOW: Warning
        LIGHTRED: ERROR
        RED: CRITICAL 
    
    """
    DEBUG = "\033[92m"  # LIGHT GREEN
    INFO = "\033[94m"   # LIGHT BLUE
    # DEBUG = '\033[32m'    # GREEN
    # INFO = "\033[97m"     # BLUE
    WARNING = "\033[33m"
    ERROR = '\033[91m'
    CRITICAL = "\033[31m"

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MyLogger:    
    """Allow users to easily log messages to a common location with a class method, from any file
    
    Class Variables:
        loggerNames (dict):  Keeps track of all our predefined logging variables
        c_handler (StreamHandler): Console handler for the logging class
        h_handler (FileHandler): File handler for the logging class
        f_handler_comps (StreamHandler): Log search objects for post-processing and plotting

    Levels:
        Debug:
        Info:
        Warning:
        Error:
        Critical:

    Todo:
        * Figure how to pass multiple arguments to string format

    """
    # Tracker: track all predefined objects, to reuse them
    loggerNames = dict()

    # create console handlers based on global cfg
    c_handler = logging.StreamHandler()
    c_handler.setLevel(eval("logging.{}".format(cfg.Misc.console_level.upper())))
    
    # create console formatters and add it handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)

    # # Now create file handlers if enabled
    # if os.path.exists(os.path.join(directory, self.filename)):
    #     raise FileExistsError('{} already exists!'.format(self.filename))
    # else:
    #     break

    # f_handler = logging.FileHandler()


    @classmethod
    def add_message(cls, msg, logger_name, input_level, args=None):
        if logger_name not in cls.loggerNames :

            # initialize
            cls.loggerNames[logger_name] = {}

            # Create a custom logger
            logger = logging.getLogger(logger_name)   

            # add handlers, add to tracker
            cls.create_handler(logger, logger_name)

        else:                      
            # get logger from tracker
            logger = cls.loggerNames[logger_name]["logger"]

        # log the message
        textColor = "Ccolors.{}".format(input_level.upper())
        textEnd = Ccolors.ENDC
        # TODO need to replace first and last double quotes: pattern = r"(.+)('\d+')(.+)"

        if args is not None:
            eval("""logger.{}("{},{}")""".format(input_level.lower(), eval(textColor) + msg + textEnd, args))
        else:
            eval("""logger.{}("{}")""".format(input_level.lower(), eval(textColor) + msg + textEnd))

    @classmethod
    def create_handler(cls, logger, logger_name):
        # Add to our tracker
        # cls.loggerNames[logger_name]["handler"] = cls.c_handler
        # cls.loggerNames[logger_name]["handler_format_string"] = "logging.{}".format(input_level.upper())

        # Add handlers to logger (to filter)
        logger.addHandler(cls.c_handler)

        # Log everything
        # https://stackoverflow.com/questions/47590989/what-is-the-difference-between-java-util-logger-setlevel-and-handler-setlevel
        logger.setLevel(logging.DEBUG)

        # add logger instance to tracker
        cls.loggerNames[logger_name]["logger"] = logger



        
        

        
