import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest

from data.stream import Stream
from data.video import VideoStream

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

    def test_batch_basics(self):
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

    def test_map(self):
        # TODO: More test cases
        simple_stream = Stream(iter(self.simple_list))
        map_fn = lambda x: x + 30
        mapped_list = map(map_fn, self.simple_list)
        mapped_stream = simple_stream.map([map_fn])
        self.assertEqual(mapped_list[0], mapped_stream.next())
        self.assertEqual(mapped_list[1:], mapped_stream.to_list())
        # Check map ordering is correct
        map_fns = [map_fn, lambda x: x / 2.0]
        mapped01 = map(map_fns[1], map(map_fns[0], self.simple_list))
        mapped10 = map(map_fns[0], map(map_fns[1], self.simple_list))
        # Reset simple_stream
        simple_stream = Stream(iter(self.simple_list))
        mapped_stream01 = simple_stream.map(map_fns)
        self.assertEqual(mapped01, mapped_stream01.to_list())
        # Reset simple_stream
        simple_stream = Stream(iter(self.simple_list))
        mapped_stream10 = simple_stream.map(map_fns[::-1])
        self.assertEqual(mapped10, mapped_stream10.to_list())

    def test_reduce(self):
        # TODO: More test cases
        simple_stream = Stream(iter(self.simple_list))
        list_sum = reduce(lambda x,y: x+y, self.simple_list)
        self.assertEqual(list_sum, simple_stream.reduce(lambda x,y: x+y))

class TestVideoStream(unittest.TestCase):

    def setUp(self):
        self.filename = 'test_video.avi'

    def test_init(self):
        vs = VideoStream(self.filename)
        self.assertTrue(vs.fps > 0)
        self.assertEqual(512, vs.width)
        self.assertEqual(512, vs.height)

    def test_save(self):
        out_name = 'test_out.avi'
        vs = VideoStream(self.filename)
        vs.save(out_name)
        vs2 = VideoStream(out_name) 
        self.assertEqual(vs.to_list(), vs2.to_list())

if __name__ == '__main__':
    unittest.main()
