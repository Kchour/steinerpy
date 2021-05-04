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

## Optional

There is also support for large files (i.e. results_*.pkl) using [git-lfs](https://github.com/git-lfs/git-lfs/wiki/Installation). Large files are kept in a separate server, while the repo contains only pointers to them.

If git-lfs is installed and you DO NOT wish to download any large files, run
`GIT_LFS_SKIP_SMUDGE=1 git clone SERVER-REPOSITORY`

# How to use

First import some relevant modules 
```
from steinerpy.library.graphs.graph import GraphFactory
from steinerpy import Context
```

Create a squareGrid
```
# Spec out our `SquareGrid` type graph object using `GraphFactory`
minX = -15			# [m]
maxX = 15           
minY = -15
maxY = 15
grid = None         # pre-existing 2d numpy array?
grid_size = 1       # grid fineness[m]
grid_dim = [minX, maxX, minY, maxY]
n_type = 8           # neighbor type

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

Currently supported inputs are `Astar`, `SstarAstar`, `SstarDijkstra`.

See `Results` and `Tests` folder for more examples

# TODO List
See todo list

# Running tests
Make sure the global config `visualize` is set to `False`. In the future, we will find a better way to set global flags

## Single test
Go to the `tests` folder in the root directory, and run any of the scripts there using `python3 script.py`.

## Automated tests
cd in root directory, then run `python3 -m unittest discover`. By default `discover` tries to look for \*tests\*, but can be changed using `-p, --pattern` flag

# Misc Notes
- Make sure to unset `PYTHONPATH` before creating a virtual environment test
    ```
    $ Unset PYTHONPATH
    $ python3 -m venv test_venv
    $ source test_venv/bin/activate
    ```

Note that this will only affect the current shell and is not permanent!
    
- After building docs, we can start an http server and open `index.html`

    ```
    $ python -m http.server
    ```

Then in the browser open, whatever ip address is displayed in the shell

- Build docs with `sphinx` command
    ```
    $ cd docs
    $ sphinx-quickstart # not necessary for us 
    $ make html
    ```
- git-lfs tutorial: [link](https://github.com/git-lfs/git-lfs/wiki/Tutorial)
- To use git-lfs for the first time after a large file has been previously committed [link](https://github.com/git-lfs/git-lfs/issues/1328)
    - back up current branch
    - use `git reset {sha}` to find last parent without issues
    - track large files with `git-lfs` using `.gitattributes`
    - commit and push
- To ignore lfs file downloads globally: `git lfs install --skip-smudge`
