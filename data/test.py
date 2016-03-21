import unittest

from stream import Stream

class TestStream(unittest.TestCase):

    def setUp(self):
        self.simple_list = range(10)
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

if __name__ == '__main__':
    unittest.main()
