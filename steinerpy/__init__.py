import logging, logging.config

# add nullhandler to a logger named after our library
logging.getLogger('steinerpy').addHandler(logging.NullHandler())

def enable_logger():
    """Call this load the default config 

    set level using the "set_level" function below

    """
    from .config import log_conf
    logging.config.dictConfig(log_conf)
    return logging.getLogger("steinerpy")

# convenience code so user doesn't need to import logging
DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARN
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

def set_level(level):
    """equivalent to setting numerical level. see logging library"""
    l = logging.getLogger('steinerpy')
    l.setLevel(level)

#try a different backend?
# import matplotlib
# # # matplotlib.use('Agg')
# # matplotlib.use('Qt5Agg')
# matplotlib.use('Qt4Agg')
# matplotlib.interactive(True)