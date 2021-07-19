from steinerpy.library.graphs.graph import GraphFactory
from steinerpy.context import Context

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

terminals = [(-10, -12), (-3, 10), (10, -7), (13, 6), (0, 3)]

# Create context
context = Context(graph, terminals)

# run and store results for astar
context.run('S*-HS')
results = context.return_solutions()


