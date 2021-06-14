# Installation
Requires Python 3.6 or higher (lower version not guaranteed to work)

First, clone the repository and install requirements 
```
pip3 install -r requirements.txt
```

Then use pip3 to install this package
```
pip3 install steinerpy
```

For developers, install the project in editable mode [more info](https://stackoverflow.com/questions/60638356/difference-between-pip-install-and-pip-install-e)

```
pip3 install -e .
```
# Getting Started

First import some relevant modules 
```
from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.context import Context
```

Create a squareGrid
```
# Spec out our `SquareGrid` type graph object using `GraphFactory`
minX = -15			# [m]
maxX = 15           
minY = -15
maxY = 15
grid = None         # pre-existing 2d numpy array, where 1=obstacle
grid_size = 1       # grid fineness
grid_dim = [minX, maxX, minY, maxY]
n_type = 8          # either 8 or 4 cell neighbors

# Create a squareGrid using GraphFactory
graph = GraphFactory.create_graph("SquareGrid", grid=grid, grid_dim=grid_dim, grid_size=grid_size, n_type= n_type)  
```

Define some terminals
```
terminals = [(-10, -12), (-3, 10), (10, -7), (13, 6), (0, 3)]
```

Now run an algorithm using the `Context` class:
```
# Create context
context = Context(graph, terminals)

# run and store results for astar
context.run('Astar')
results = context.return_solutions()
```

`results` will look like the following:
```
>>> print(results)
{'dist': [8.242640687119284, 14.142135623730951, 14.242640687119284, 19.14213562373095], 'path': [array([[-3, 10],
   ...[ 0,  3]]), array([[ 0,  3],
   ...[10, -7]]), array([[ 0,  3],
   ...[13,  6]]), array([[  0,   3],
 ...10, -12]])], 'sol': [(...), (...), (...), (...)]}
```

Currently supported inputs are `S*-unmerged`, `S*-HS`, `S*-HS0`, `Kruskal`, `S*-BS`, `S*-MM`, and `S*-MM0`.

See `Tests` folder for more examples

# Configuring Behavior

The `Config` module is provided for configuring global behavior. Simply import it before all other `steinerpy` packages and directly modify class variables, e.g. to visualize the algorithms as they run,

```
import steinerpy.config as cfg
cfg.Animation.visualize = True
```

or to adjust heuristics

```
import steinerpy.config as cfg
cfg.Algorithm.sstar_heuristic_type = diagonal_uniform # This is the default option
cfg.Algorithm.hFactor = 1.0  # Heuristic factor i.e. for any node n, its priority is given by f(n) = g(n) + hFactor*h(n)
```

See the config module for more options (note: not everything has been implemented)

# Customize the Heuristic Function
The underlying heuristic in all heuristic-based algorithms can be customized by binding `steinerpy.algorithms.common.custom_heuristics` with a function in the form `h_func(n, goal)` i.e.

```
from steinerpy.library.config import Config as cfg
cfg.sstar_heuristic_type = "custom"

from steinerpy.algorithms.common import Common
def my_cust_h_func(n, goal):
    ...

common.custom_heuristics = my_cust_h_func
```

# Running tests
Make sure the global config `visualize` is set to `False`. In the future, we will find a better way to set global flags

## Single test
Go to the `tests` folder in the root directory, and run any of the scripts there using `python3 script.py`.

## Automated tests
cd in root directory, then run `python3 -m unittest discover`. By default `discover` tries to look for \*tests\*, but can be changed using `-p, --pattern` flag

# TODO List
See todo list (very wip)
