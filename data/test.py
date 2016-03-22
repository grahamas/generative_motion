import unittest

from stream import Stream

class TestStream(unittest.TestCase):

    def setUp(self):
        self.LIST_LEN = 20
        self.simple_list = range(self.LIST_LEN)
    def test_init(self):
        simple_stream = Stream(iter(self.simple_list))
        LIMIT = 5
        simple_stream_limited = Stream(iter(self.simple_list), max_len=LIMIT)
        SKIP = 3
        simple_stream_skipped = Stream(iter(self.simple_list), skip_len=SKIP, max_len=LIMIT)
        SHIFT = 80
        simple_stream_shifted = Stream(iter(self.simple_list), on_load=[lambda x: x+SHIFT])

        self.assertEqual(self.simple_list, simple_stream.to_list())
        self.assertEqual(self.simple_list[:LIMIT], simple_stream_limited.to_list())
        self.assertEqual(self.simple_list[SKIP:SKIP+LIMIT], simple_stream_skipped.to_list())
        self.assertEqual([x + SHIFT for x in self.simple_list], simple_stream_shifted.to_list())

    def test_chaining(self):
        simple_stream = Stream(iter(self.simple_list))
        chain_stream = Stream(simple_stream)
        self.assertEqual(self.simple_list[0], chain_stream.next())
        # TODO: Decide if this is proper behavior.
        self.assertEqual(self.simple_list[1], simple_stream.next()) 
        self.assertEqual(self.simple_list[2], chain_stream.next())

    def test_batch(self):
        BATCH_SIZE = 2
        simple_stream = Stream(iter(self.simple_list))
        simple_batch_stream = simple_stream.batch(BATCH_SIZE)
        self.assertEqual(self.simple_list[:BATCH_SIZE], simple_batch_stream.next())
        self.assertEqual(self.simple_list[BATCH_SIZE:BATCH_SIZE*2], simple_batch_stream.next())
        chained_batch_stream = Stream(simple_batch_stream)
        self.assertEqual(self.simple_list[BATCH_SIZE*2:BATCH_SIZE*3], chained_batch_stream.next())
        double_batch_stream = chained_batch_stream.batch(2)
        self.assertEqual([self.simple_list[BATCH_SIZE*3:BATCH_SIZE*4], self.simple_list[BATCH_SIZE*4:BATCH_SIZE*5]],
                double_batch_stream.next())
        # Change vars and reset simple_stream
        STRIDE_SIZE = 1
        BATCH_SIZE = 3
        simple_stream = Stream(iter(self.simple_list))
        stride_batch_stream = simple_stream.batch(BATCH_SIZE, stride=STRIDE_SIZE)
        self.assertEqual(self.simple_list[:BATCH_SIZE], stride_batch_stream.next())
        self.assertEqual(self.simple_list[0+STRIDE_SIZE:BATCH_SIZE+STRIDE_SIZE], stride_batch_stream.next())
        # Reset simple_stream again
        simple_stream = Stream(iter(self.simple_list))
        whole_batch_stream = simple_stream.batch(self.LIST_LEN)
        self.assertEqual(self.simple_list, whole_batch_stream.next())
        self.assertIsNone(whole_batch_stream.next())
        # Reset simple_stream again
        simple_stream = Stream(iter(self.simple_list))
        long_batch_stream = simple_stream.batch(self.LIST_LEN-1, stride=1)
        self.assertEqual(self.simple_list[:-1], long_batch_stream.next())
        self.assertEqual(self.simple_list[1:], long_batch_stream.next())
        self.assertIsNone(long_batch_stream.next())



class TestBatchStream(unittest.TestCase):

    def setUp(self):
        self.LIST_LEN = 20
        self.simple_list = range(self.LIST_LEN)
    def test_init(self):
        pass

if __name__ == '__main__':
    unittest.main()
