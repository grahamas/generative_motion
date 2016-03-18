import os,sys

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

import stream

FLOAT_STR = 'float64'
theano.config.floatX = FLOAT_STR
np_floatX = np.dtype(FLOAT_STR)

class MovieCollection(object):
    """
        Maintains a collection of movies for training.
        On every iteration, loads a certain number of frames from every
        movie in the collection into an array for consumption by model.
    """
    def __init__(self, movie_args, frame_dims, frame_step_size=1000,
            movie_step_size=8):
        self.movie_args = movie_args
        self.frame_dims = frame_dims
        self.frame_step_size = frame_step_size
        self.movie_step_size = movie_step_size

        step_arr = np.zeros([self.movie_step_size * self.frame_step_size] + self.frame_dims,
            dtype=np_floatX)

        self.movies = [MovieStream.from_args(args) for args in self.movie_args] 

    @classmethod
    def from_json(cls, fp_json, step_size=None):
        with open(fp_json, 'r') as f_json:
            dct_json = json.load(f_json)
        frame_dims = dct_json['frame_dims']
        if 'step_size' in dct_json and not step_size:
            step_size = dct_json['step_size']
        movies_path = dct_json['movies_path']
        movie_args = dct_json['movie_args']
        for i_arg in range(len(movie_args)):
            movie_args[i_arg]['fn'] = os.path.join(movies_path, movie_args[i_arg]['fn'])
        if step_size:
            return cls(movie_args, frame_dims, step_size)
        else:
            return cls(movie_args, frame_dims)

    def __iter__(self):
        self.movie_iters = map(lambda x: x.__iter__(), self.movies) 
        # TODO: Evaluate whether or not to include "viable_movies"
        #self.viable_movies = range(len(self.movies))
        return self
    def next(self):
        """
            Fills step_arr with new movies (pseudorandomly, but in order)
            Currently STOPS when any movie runs out. I figure this is better
            than the other trivial option of stopping when all movies run out,
            so that one training set doesn't dominate.
        """
        i_arr = 0
        prev_start = 0
        movie_ends = []
        movie_nums = []
        for i_movie in range(self.movie_step_size):
            i_rand_movie = random.randrange(len(self.movies))
            movie_iter = self.movie_iters[i_rand_movie]
            n_frames_loaded = movie_iter.load_array(self.step_arr[i_arr:i_arr+self.frame_step_size,:,:], self.frame_step_size)
            if n_frames_loaded < self.frame_step_size:
                raise StopIteration
            prev_start += n_frames_loaded
            movie_ends += prev_start
            movie_nums += i_rand_movie
        return np.vstack([movie_nums,movie_ends])

class MovieStream(stream.FileStream):
    """
        Streams a movie from a VideoCapture.
        Does not hold movie; IS ITERATOR.
    """
    def __init__(self, file_path, max_len=None,
        skip_len=0, on_load=[],
        fps=None, grayscale=True):
        """
            file_path       : path to .avi file
            n_frames        : (optional, defaults to all) number of frames to load
            drop_frames     : (optional) the number of initial frames to drop
            preprocessing   : (default=[]) list of preprocessing methods 
                                (Use grayscale option for grayscale preprocessing)
            grayscale       : (default=True) must specify for loading+saving
        """
        super(MovieStream, self).__init__(file_path, max_len, skip_len, on_load)
        # We open the file to get metadata
        self._cap = None
        self._open_cap()
        # Get the number of movie frames and make sure it is 
        # commensurate with the requested n_frames and drop_frames
        # NOTE: Currently this doesn't work on midway (using old OpenCV)
        total_n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert total_n_frames >= max_len + skip_len, 
             "{} < {} + {}".format(total_n_frames, 
             max_len, skip_len)
        # Get the movie dimensions
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self._cap.get(cv2.CAP_PROP_FPS)
        if fps:
            assert self.fps == fps
        self._close_cap()
        if n_frames == 0:
            self.n_frames = sys.maxint - self._drop_frames
        else:
            self.n_frames = n_frames
        self._preprocessing = preprocessing
        self.grayscale = grayscale
        if grayscale:
            self._preprocessing.insert(0,vu.RGB_to_grayscale)
    @classmethod
    def from_args(cls, args):
        return cls(args['fn'])
    def next(self):
        """
            Iterates while cap remains opened and we haven't advanced beyond n_frames.
            Applies pre-processing at every call. 
        """
        if self._cap.isOpened() and ((not self.n_frames) or (self.i_frame < self.n_frames + self._drop_frames)):
            ret, frame = self._cap.read()
            if frame is None:
                raise StopIteration
            processed = reduce(lambda x,f: f(x), self._preprocessing, frame)
            self.i_frame += 1
            return processed
        else:
            self._close_cap()
            raise StopIteration
    def __enter__(self):
        """
            Defines and instantiates MovieSource subclass of FileSource
            on entrance into "with" context manager.

            Allows ensured clean-up by __exit__
        """
        class MovieSource(stream.FileSource):
            def __init__(self, file_path, max_len, skip_len):
                self.file_path = file_path
                self.max_len = max_len
                self.skip_len = skip_len
                cap = VideoCapture(os.path.realpath(self.file_path))
                if not cap.isOpened():
                    raise ValueError("Cannot open movie file path.")
                cap.skip_to_frame(0) # Skips initially skipped frames (skip_len)
                self.i_frame = 0
                self.cap = cap
            def close_file(self):
                if self.cap:
                    self.cap.release()
                    self.i_frame = None
            def skip(self):
                self.skip_to_frame(self.i_frame + 1)
            def skip_n(self, n):
                self.skip_to_frame(self.i_frame + n)
            def skip_to_frame(self, i_frame):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame + self.skip_len)
                self.i_frame = i_frame
            def next(self):
                if self.cap.isOpened() and ((not self.max_len) or (self.i_frame < self.max_len)):
                    return self.cap.read()
                else:
                    return None
        self.source = MovieSource(self.file_path)
        return self.source
    def __exit__(self):
        self.source.close_file()
    def play(self, window_name='frame'):
        """
            Play the movie using OpenCV.
        """
        frame_len = int(1000.0 / self.fps)
        with self as stream:
            frame = stream.next()
            while frame is not None:
                cv2.imshow(window_name, np.array(np.squeeze(frame), dtype=np.uint8))
                if cv2.waitKey(frame_len) & 0xFF == ord('q'):
                    break
                frame = stream.next()
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
    def save(self, file_name):
        """
            Save the movie to file_name as XVID avi
        """
        writer = VideoWriter(file_name, frameSize=(self.width, self.height),
                fourcc='XVID')
        writer.open()
        if not writer.isOpened():
            raise Exception("Did not open movie file for writing.")
        for frame in self:
            if self.grayscale:
                frame = vu.grayscale_to_RGB(frame)
            writer.write(frame)
        writer.release()
    def average_luminance(self):
        """
            Return array of average frame luminances.
        """
        return np.array([np.mean(frame) for frame in self])
    def luminance_spectrum(self, smooth=None):
        """
            Periodogram of average luminance, with optional smoothing.
        """
        avg_lum = self.average_luminance()
        f, Pxx =  signal.periodogram(avg_lum, self.fps)
        if smooth:
            Pxx = smooth(Pxx)
            if len(Pxx) < len(f):
                # If smoothing shrinks Pxx, then shrink f
                f = f[-len(Pxx):]
        return f, Pxx

