import unittest
from steinerpy.environment import EnvType, EnvLoader

class TestLoadDiffEnvironments(unittest.TestCase):

    def test_load_mapf(self):
        name = "Berlin_1_256.map"
        graph = EnvLoader.load(EnvType.MAPF, name)

    def test_load_3d_grid(self):
        name = "A1.3dmap"
        graph = EnvLoader.load(EnvType.GRID_3D, name)

        # get neighbors (0,0,0)
        res = list(graph.neighbors((0,0,0)))
        self.assertTrue(len(res), 7)

        res = list(graph.neighbors((1,1,1)))
        self.assertTrue(len(res), 26)

        res = graph.cost((0,0,0), (1,1,1))
        self.assertTrue( abs(res - 1.7320508075688772)<=1e-6)

        # slow with matplotlib right now
        # graph.show_grid()

    def test_load_2d_grid(self):
        name = "sc/WheelofWar.map"
        graph = EnvLoader.load(EnvType.GRID_2D, name)
        # graph.show_grid()

if __name__ == "__main__":
    unittest.main(verbosity=2)