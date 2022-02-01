"""Basic heuristics implemented here for 2D grids


"""
import numpy as np
import steinerpy.config as cfg

class Heuristics:
    @staticmethod
    def heuristic_func_wrap(*args, **kwargs)->float:
        if cfg.Algorithm.graph_domain == "grid":
            return _GridBasedHeuristics2D.mapper[cfg.Algorithm.sstar_heuristic_type](*args, **kwargs)
        elif cfg.Algorithm.graph_domain == "generic":
            return _CustomHeuristics.h_func(*args, **kwargs)
        else:
            raise ValueError("graph_domain needs to be either the following: {}, {}".format('\'grid\'', '\'generic\''))
    
    @staticmethod
    def bind(func):
        """Rebind custom heuristic function
            and also update the mapping!
        """
        _CustomHeuristics.h_func = func

        # the 2d class will also refer to custom heuristics
        _GridBasedHeuristics2D.mapper["custom"] = func

    @staticmethod
    def get():
        return _CustomHeuristics.h_func

class _CustomHeuristics:
    """Allow users to specify their own heuristic func by re-binding 
    from the function above

    """
    @staticmethod
    def h_func(next, goal):
        """The heuristic function take in two params and return a float. This can
            be rebinded using the function `bind` function below
        
        Parameters:
            next (tuple):
            goal (tuple):

        """
        raise ValueError("User needs to specify a heuristic function, i.e. call {}".format("CustomHeuristics.bind(lambda next,goal: 0)"))

# constants for voxel heuristic 
vl = 1
SQ3 = 1.7320508075688772
SQ2 = 1.4142135623730951
C1 = SQ3 - SQ2
C2 = SQ2 - vl
C3 = vl

class _GridBasedHeuristics2D(type):
    """
    Heuristics for a flat grid graph

    Parameters:
        next (tuple): The source vertex  
        goal (tuple): The destination vertex

    """

    def __init__(cls): 
    
        cls.mapper = {"manhattan": _GridBasedHeuristics2D._gbh_manhattan,
                    "zero": _GridBasedHeuristics2D._gbh_zero,
                    "euclidean": _GridBasedHeuristics2D._gbh_euclidean,
                    "diagonal_uniform": _GridBasedHeuristics2D._gbh_diagonal_uniform,
                    "diagonal_nonuniform": _GridBasedHeuristics2D._gbh_diagonal_nonuniform,
                    "preprocess": _GridBasedHeuristics2D._gbh_preprocess,
                    "custom": _CustomHeuristics.h_func,
                    "voxel": _GridBasedHeuristics2D._gbh_voxel}
    @staticmethod
    def _gbh_manhattan(cls, next: tuple, goal: tuple):
        (x1, y1) = next
        (x2, y2) = goal
        return  abs(x1 - x2) + abs(y1 - y2)

    @staticmethod
    def _gbh_euclidean(next, goal):
        (x1, y1) = next
        (x2, y2) = goal
        v = [x2 - x1, y2 - y1]
        return np.hypot(v[0], v[1])

    @staticmethod
    def _gbh_diagonal_uniform(next, goal):
        (x1, y1) = next
        (x2, y2) = goal
        return max(abs(x1 - x2), abs(y1 - y2))
    
    @staticmethod
    def _gbh_diagonal_nonuniform(next, goal):
        (x1, y1) = next
        (x2, y2) = goal
        dmax = max(abs(x1 - x2), abs(y1 - y2))
        dmin = min(abs(x1 - x2), abs(y1 - y2))
        return 1.414*dmin + (dmax - dmin)


    @staticmethod
    def _gbh_voxel(next, goal):
        x1, y1, z1 = next 
        x2, y2, z2 = goal
        dx, dy, dz = abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)
        dmax = max(dx, dy, dz)
        dmin = min(dx, dy, dz)
        dmid = dx + dy + dz - dmin - dmax 
        return C1*dmin + C2*dmid + C3*dmax        

    @staticmethod
    def _gbh_preprocess(next, goal):
        """Get heuristic value from pre-processed lookup table"""
        from steinerpy.library.pipeline import GenerateHeuristics
        return GenerateHeuristics.heuristic_wrap(next, goal)    

    @staticmethod
    def _gbh_zero(next, goal):
        return 0

# initialize
_GridBasedHeuristics2D.__init__(_GridBasedHeuristics2D)
