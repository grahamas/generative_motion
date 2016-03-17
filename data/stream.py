import os,sys

from abc import ABCMeta, abstractmethod

import random

import numpy as np
from scipy import signal

import json
import video as vu
#import cv2
from skvideo.io import VideoCapture, VideoWriter
#from cv2 import VideoCapture, VideoWriter
import skimage.io as skio

import theano

import matplotlib.pyplot as plt

FLOAT_STR = 'float64'
theano.config.floatX = FLOAT_STR
np_floatX = np.dtype(FLOAT_STR)

class Stream(object):
    __metaclass__ = ABCMeta
    """
        Streams data
        Does not hold data; IS ITERATOR.
    """
    def __init__(self, source, max_len=None,
        skip_len=0, on_load=[]):
        """
            source          : stream's source
            max_len         : (default=None) maximum number of stream units to load
            skip_len        : (default=0) the number of initial units to skip
            on_load         : (default=[]) list of processing functions to apply to 
                                each unit when loaded
        """
        self.source = source
        self.max_len = max_len
        self.skip_len = skip_len
        self.on_load = on_load

    def write(self, file_name, on_write=[], mode='wb'):
        with open(file_name, mode) as out_file:
            for unit in self:
                out_file.write(reduce(lambda x,f: f(x), on_write, unit))

    def next(self):
        """
            Returns next unit.
        """
        return reduce(lambda x,f: f(x), self.on_load, source())
    @abstractmethod
    def batch(self, batch_size, stride):
        """
            Returns stream whose units are batches.
        """
        pass
    @abstractmethod
    def map(self, functions, arr=None):
        """
            Returns stream whose units are map output.
        """
        pass
    @abstractmethod
    def reduce(self, function, arr=None):
        """
            Returns output of reduction.
        """
        pass

class FileStream(Stream):
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

class FileSource(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def next(self):
        pass
    @abstractmethod
    def close_file(self):
        pass

