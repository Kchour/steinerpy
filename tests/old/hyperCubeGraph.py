import itertools as it


class HyperCubeGraph:
    
    def __init__(self, k: int, hamm_dist):
        # dimension of hypercube
        self.k = k 
        self.hamm_dist = hamm_dist
        self.bitmasks = [n for n in range(k)]

        self.visited = set()

    def neighbors(self, v:int):
        assert (v<2**self.k)
        neighs = []
        for b in it.combinations(self.bitmasks, 2):
            # apply bit masks
            temp = v
            for bb in b:
                temp ^= 2**bb
            neighs.append(temp)

        return neighs

    def cost(self):
        pass

    def node_count(self):
        pass

    def edge_count(self):
        pass

    def dfs(self, v:int, init=True):
        # """Runs into max recursion limit"""
        # if init:
        #     self.visited = set()

        # if v not in self.visited:
        #     self.visited.add(v)
        #     for n in self.neighbors(v):
                
        #         self.dfs(n, init=False)

        q = [v]
        while q:
            v = q.pop()

            if v not in self.visited:
                self.visited.add(v)
                for n in self.neighbors(v):

                    q.append(n)  
        
if __name__ == "__main__":    
    hg = HyperCubeGraph(5, 2)

    # check neighbors
    neighs= hg.neighbors(0)

    # print(bin(0))
    # for n in neighs:
    #     print(bin(n))

    # check dfs
    hg.dfs(0)
    print(hg.visited)

    # convert to bin
    print([bin(b) for b in hg.visited])