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
        Streams a data file
        Does not hold data; IS ITERATOR.
    """
    def __init__(self, file_path, max_len=None,
        skip_len=0, on_load=[]):
        """
            file_path       : path to data file
            max_len         : (default=None) maximum number of stream units to load
            skip_len        : (default=0) the number of initial units to skip
            on_load         : (default=[]) list of processing functions to apply to 
                                each unit when loaded
        """
        self.file_path = file_path
        self.max_len = max_len
        self.skip_len = skip_len
        self.on_load = on_load

    @abstractmethod
    def __iter__(self):
        pass
    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def fill_array(self, arr):
        pass

