import unittest
from circ_array import CircularArray

class TestCircArray(unittest.TestCase):

    def test_init_get(self):
        self.assertEqual(range(5), CircularArray(range(5)).get())
        self.assertEqual([], CircularArray([]).get())
    def test_len(self):
        self.assertEqual(len(range(5)), len(CircularArray(range(5))))
        self.assertEqual(len([]), len(CircularArray([])))
    def test_append(self):
        circ_range = CircularArray(range(5))
        self.assertEqual(range(20,25), circ_range.append(range(20,25)).get(), msg="with internals {0}".format(str(circ_range.array)))
        self.assertEqual([23, 24, 5, 6, 7], circ_range.append([5,6,7]).get())
        self.assertEqual(len(range(5)), len(circ_range))
        self.assertEqual([7] + range(4), circ_range.append(range(4)).get())
        self.assertEqual(range(4)[1:] + [20, 30], circ_range.append([20, 30]).get())
        self.assertEqual(range(4)[3:] + [20, 30] + [80, 90], circ_range.append([80, 90]).get())
        self.assertEqual(range(5), circ_range.append(range(5)).get(), msg="with internals {0}".format(str(circ_range.array)))
        self.assertRaises(NotImplementedError, CircularArray([]).append, [5,6])
        
if __name__ == '__main__':
    unittest.main()
