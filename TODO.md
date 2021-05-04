# TODO List (wip2)
## Roadmap 1
- Generalize `search_algorithm` so user can define heuristic, g function
- Add a `custom_functions` file which allows user to define their functions to perform on the openlist, closed list, etc...
- In `main.py`, 
    - allow user to define h, g functions, which get passed to `search_algorithm`
    - Follow slides closely and define functions the profs want:
        1) nominate func (apply search algorithm for one iteration)
        2) grab subset of nodes using some criteria (user defined). Add edges to forest F
        (Need a way to "unmodify" step 1)
        3) Determine shortest paths in forest F
        4) UpdateSolutionS
- In h,g functions, figure out parameters the user has access to 
- consider keying working forests F with terminal

## Roadmap 2
- Reduce dependence on external libraries, stick to default python data types for the most part
- Add default options to Animate class (keyword arguments)
- Add option to disable early exit? (basically just don't specify goal or use `None`)
- Stop converting data types too often. Leads to confusion
- Figure out a clean way to do comments ''' vs #
- Improve naming of variables

## Roadmap 3
- Finished primal-dual! But need to refactor it
- After finishing tree, output steiner tree lengths and time?
- Implement a smarter cycle checking feature instead of DepthFirstSearch
- Need a logging feature to determine number of nodes expanded
- Explore why priorityQueue with heapq library doesn't work the same way as my own
- Need to get gVal of path in mm
- Add early exit flag (very optional atm)
- selData shouldn't need selNode in the dictionary

## Roadmap 4
- Need to add a simple way to keep track of number of operations...
- Need to return steiner tree value for each algorithm

## Roadmap 5
- Create test suite (https://stackoverflow.com/questions/31556718/is-it-possible-to-run-all-unit-test?lq=1)
- Consider creating an 'algorithms' helper class, which imports algs (i.e. astar, mm, kruskals, etc...). Then future programs that import this can be done as such:
 'from steinerpy.steiner.algorithms import blah'
- need a way to log data (using logging module or class variables)
- need a way to save/reload test terminals. Maybe pickle? 
- Generate 20 random test cases with kruskals algorithm offline. Then compare with other algorithms
- Need to have universal getter/setter for solutions
- Refine solution set S keys (path, sol, dist)

## Roadmap 6
- Allow **kwargs or *args in Framework functions, otherwise too many global variables are being passed to each other
- ABSOLUTELY NEED TO GET RID OF THESE GLOBAL VARIABLES -> LEAD TO TIGHT COUPLING BETWEEN FUNCTIONS
- When ready, move baseline.pkl and testers to /tests to do automated testing in the future
- Create a commons module to share functions between algorithms

## Roadmap 7
- Change t1,t2 in `Common` and `Framework` class description (they are not terminals, but components!)

- bootstrapping initialization and discover [link](https://stackoverflow.com/questions/7432359/bootstrapping-tests-and-using-python-test-discovery)
    provide an executable script `runtest.py` which adds root to project directory as such
    ```
    import sys, os

    sys.path.insert(0, os.path.dirname(__file__))
    ```
- Check monotonicity

