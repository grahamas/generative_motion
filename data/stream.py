import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from future_builtins import zip

from abc import ABCMeta, abstractmethod

import random

from util.circ_array import CircularArray

import numpy as np

class Stream(object):
    """
        Streams data
        Does not hold data; IS NOT ITERATOR.
        Does iterate.
    """
    def __init__(self, source, max_len=None,
        skip_len=0, on_load=[]):
        """
            source          : stream's source (typically another stream)
            max_len         : (default=None) maximum number of stream units to load
            skip_len        : (default=0) the number of initial units to skip
            on_load         : (default=[]) list of processing functions to apply to 
                                each unit when loaded
        """
        self.source = source
        self.max_len = max_len
        self.skip_len = skip_len
        self.on_load = on_load
        self.skip_n(self.skip_len)
        self.i_unit = 0

    def write(self, file_name, mode, on_write=[]):
        with open(file_name, mode) as out_file:
            unit = self.next()
            while unit is not None:
                out_file.write(reduce(lambda x,f: f(x), on_write, unit))
                unit = self.next()
    def next(self):
        """
            Returns next unit.
        """
        if (not self.max_len) or self.i_unit < self.max_len:
            try:
                self.i_unit += 1 # Probably bad idea, since StopIteration could be sent,
                                 # but i_unit would continue incrementing.
                return reduce(lambda x,f: f(x), self.on_load, self.source.next())
            except StopIteration:
                return None
        else:
            return None
    def skip(self):
        if isinstance(self.source, Stream):
            self.source.skip()
        else:
            self.source.next()
    def batch(self, batch_size, stride=None, max_len=None,
            skip_len=0, on_load=[]):
        """
            Returns stream whose units are batches.
        """
        return BatchStream(self, batch_size, stride=stride,
                max_len=max_len, skip_len=skip_len, on_load=on_load)
    def map(self, functions, arr=None, stride=None,
            max_len=None, skip_len=0):
        """
            Returns stream whose units are map output.
        """
        if isinstance(functions, list):
            return Stream(self, max_len=max_len,
                    skip_len=skip_len, on_load=functions)
        else:
            return Stream(self, max_len=max_len,
                    skip_len=skip_len, on_load=[functions])
    def reduce(self, function, init_value=None):
        """
            Returns output of reduction.
        """
        if init_value:
            left = init_value
        else:
            left = self.next()
        right = self.next()
        while right is not None:
            left = function(left, right)
            right = self.next()
        return left 
    def next_n(self, n):
        return [self.next() for i in range(n)]
    def skip_n(self,n):
        """
            Skips ahead n units, sending logic down the line.
        """
        if isinstance(self.source, Stream):
            self.source.skip_n(n)
        else:
            for i in range(n):
                self.source.next()
    def to_list(self):
        ret_list = []
        next_unit = self.next()
        while next_unit is not None:
            ret_list += [next_unit]
            next_unit = self.next()
        return ret_list


class StreamCollection(object):
    """
        A collection of streams. We assume we start with the sources.
    """
    def __init__(self, stream_collection, on_load=[]):
        self.stream_collection = stream_collection
    def __len__(self):
        return len(self.stream_collection)
    def next(self):
        return [reduce(lambda x,f: f(x), self.on_load, stream) for stream in self.stream_collection.next()]
    def skip(self):
        self.stream_collection.skip()
    def next_n(self, n):
        """
            Gets list of lists of next n from each stream,
            and applies on_load functions to each unit.

            Necessary implmentation for batching.
        """
        return [[reduce(lambda x,f: f(x), self.on_load, unit) for unit in stream_n_units]
                for stream_n_units in self.stream_collection.next()]
    def skip_n(self):
        self.stream_collection.skip_n()
    def batch(self, batch_size, stride=None, max_len=None,
            skip_len=0, on_load=[]):
        return 
    def map(self, functions, stride=None,
            max_len=None, skip_len=0):
        """
            Maps functions to all streams independently.
            Works by returning a new StreamCollection with the
            functions as the on_load in the new Collection,
            and self as the source_collection of the returned
            object.

            Returns new StreamCollection.
        """
        if isinstance(functions, list):
            return StreamCollection(self, max_len=max_len,
                    skip_len=skip_len, on_load=functions)
        else:
            return StreamCollection(self, max_len=max_len,
                    skip_len=skip_len, on_load=[functions])
    def reduce(self, function, init_values=None):
        """
            Independently reduces all streams (folding from left, obviously).

            Returns list of results.
        """
        if init_values:
            if isinstance(init_values, list):
                lefts = init_values
            else:
                lefts = [init_values] * len(self.stream_collection)
        else:
            lefts = self.next()
        rights = self.next()
        while not all(rights is None):
            valid = not rights is None
            args = zip(lefts[valid], rights[valid])
            lefts = map(lambda arg: function(*arg), args)
            rights = self.next()
    def join(self, function, max_len=None,
            skip_len=0):
        """
            Joins all streams in the collection into a single stream.
            Takes a function that takes a list of units and returns 
            a unit (no necessary relation between in and out units).

            Returns new Stream.
        """
        return Stream(self, on_load=[function])


class BatchStream(Stream):
    def __init__(self, source, batch_size, 
            stride=None,
            max_len=None,
            skip_len=0, on_load=[]):
        # Notice that super call MUST follow self.next assignment,
        # in case skip function is called. (??? No that's wrong)
        if stride and not stride == batch_size: 
            assert stride < batch_size and stride > 0
            self.stride = stride
            self.next = self.uninitialized_next
        else:
            self.next = self.simple_next
        super(BatchStream, self).__init__(source, max_len, skip_len, on_load)
        self.batch_size = batch_size
    def uninitialized_next(self):
        """
            Initializes the on_hand buffer, and returns the first batch.
        """
        self.circ_array = CircularArray(self.source.next_n(self.batch_size))
        self.next = self.initialized_next
        return self.apply(self.circ_array.get())
    def initialized_next(self):
        """
            Now that the buffer has been initialized, just gets next
            stride's worth of units from source. Uses effectively circular
            array for storage.
        """
        self.circ_array.append(self.source.next_n(self.stride))
        return self.apply(self.circ_array.get())
    def simple_next(self):
        """
            In the case that the batches don't overlap.
        """
        return self.apply(self.source.next_n(self.batch_size)) 
    def apply(self, retval):
        """
            A helper function to apply on_load and check for None value.

            Possibly unnecessary, but eliminates code repetition.
        """
        if retval is None:
            return None
        retval = reduce(lambda x, f: f(x), self.on_load, retval)
        if not any(val is None for val in retval):
            return retval
        else:
            return None

class BatchStreamCollection(StreamCollection):
    def __init__(self, stream_collection, batch_size,
            stride=None,
            max_len=None,
            skip_len=0, on_load=[]):
        if stride and not stride == batch_size:
            assert stride < batch_size and stride > 0
            self.stride = stride
            self.next = self.uninitialized_next
        else:
            self.next = self.simple
        super(BatchStreamCollection, self).__init__(stream_collection, max_len, skip_len, on_load)
        self.batch_size = batch_size
    def uninitialized_next(self):
        self.l_circ_array = [CircularArray(source.next_n(self.batch_size)) for source in self.stream_collection]
        self.next = self.initialized_next
        return [reduce(lambda x,f: f(x), self.on_load, ca.get()) for ca in self.l_circ_array]
    def initialized_next(self):
        (ca.append(source.next_n(self.stride)) for ca,source in zip(self.l_circ_array,self.stream_collection))
        return [reduce(lambda x,f: f(x), self.on_load, ca.get()) for ca in self.l_circ_array]
    def simple_next(self):
        return [reduce(lambda x,f: f(x), self.on_load, source.next_n(batch_size)) for source in self.stream_collection]


class FileStream(Stream):
    """
        Abstract class for implementing base stream from a file
        Needs to be abstract as the method of file opening is different.

        In implementing, subclass FileSource in a class defined in the __enter__
        method. This subclass should define all the low level methods of interacting
        with the file. Then the __enter__ method returns an instance of the FileSource
        subclass, and the __exit__ method cleans up.
    """
    __metaclass__ = ABCMeta
    def __init__(self, file_name, max_len=None,
            skip_len=0, on_load=[]):
        """
            file_name       : path to data file
            max_len         : (default=None) maximum number of stream units to load
            skip_len        : (default=0) the number of initial units to skip
            on_load         : (default=[]) list of processing functions to apply to 
                                each unit when loaded
        """
        self.file_name = file_name
        super(FileStream, self).__init__(None, max_len,
                skip_len, on_load) 
    @abstractmethod
    def __enter__(self):
        pass
    @abstractmethod
    def __exit__(self):
        pass

class FileStreamCollection(StreamCollection):
    """
        Collection of FileStreams
        Implements "with" logic (returns list of opened sources)
    """
    def __len__(self):
        return len(self.sources)
    def next(self):
        return [reduce(lambda x,f: f(x), self.on_load, stream) for stream in self.streams]
    def skip(self):
        pass
    # TODO
    #def map(self, functions
    def __enter__(self):
        self.sources = [stream.__enter__() for stream in self.streams]
        return self.sources
    def __exit__(self):
        [source.__exit__() for source in self.sources]

class FileSource(object):
    """
        Abstract class for implementing lowest level interaction with file.
        For safety, this class should only be instantiated within a "with"
        block, so all subclass definitions should appear in the __enter__
        method of a subclass of the FileStream class.
    """
    __metaclass__ = ABCMeta
    @abstractmethod
    def next(self):
        pass
    def next_n(self,n):
        "Naive implementation, for convenience."
        return [self.next() for i in range(n)]
    @abstractmethod
    def skip(self):
        pass
    def skip_n(self,n):
        "Naive implementation, for convenience."
        for i in range(n):
            self.skip()
    @abstractmethod
    def close_file(self):
        pass

