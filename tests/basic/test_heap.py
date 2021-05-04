import unittest

from steinerpy.library.search.search_utils import PriorityQueueHeap

class TestHeap(unittest.TestCase):

    def test_priority_queue_heap(self):
        pq = PriorityQueueHeap()
        pq.put("apple", 0)
        pq.put("banana", 1)
        pq.put("celery", 2)
        pq.put("stock", 2)
        pq.put("wood", 3)
        pq.put("wood", 0)

        # Get the minimum
        min_entry = pq.get_min()

        # actually pop queue
        popped_entry = pq.get()

        self.assertEqual(pq.get_min(), (0, 'wood'))

if __name__=="__main__":
    unittest.main()

