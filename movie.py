import os

import numpy as np
from scipy import signal
import tensorflow as tf

import cv2
import video_utils as vu

import matplotlib.pyplot as plt

class Movie(object):
    """Holds a movie."""
    def __init__(self, file_path, 
        storage_array=None, buffer=None,
        n_frames=0, drop_frames=0, fps=None,
        preprocessing=[], grayscale=True):
        """
        TODO: Integrate this with Tensorflow queueing. Meaning, don't
                load the entire movie into memory.
            file_path       : absolute or relative path to .avi file
            storage_array   : (optional) preallocated storage for entire movie
            buffer          : (optional, unused) preallocated buffer. unnecessary?
            n_frames        : (optional) the number of frames to take
            drop_frames     : (optional) the number of initial frames to drop
            preprocessing   : (default=[]) list of preprocessing methods (NO GRAY)
            grayscale       : (default=True) must specify for loading+saving
        """
        self.file_path = file_path

        # We open the file to get metadata
        cap = self._open()
        # Get the number of movie frames and make sure it is 
        # commensurate with the requested n_frames and drop_frames
        total_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #assert total_n_frames >= n_frames + drop_frames
        self.drop_frames = drop_frames
        if n_frames == 0: 
            self.n_frames = total_n_frames
        else:
            assert n_frames > 0
            self.n_frames = n_frames
        # Get the movie dimensions
        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = cap.get(cv2.CAP_PROP_FPS)
        if fps:
            assert self.fps == fps
        cap.release()
        # Provide option to save movie to pre-allocated
        # storage array. Assert it is correct shape.
        if storage_array:
            self._prealloc = True
            assert np.shape(storage_array) == (self.n_frames, self.height, self.width)
            self._movie = storage_array
        else:
            self._prealloc = False
            self._movie = None # We only allocate on request. Ideally never.
        self._movie_loaded = False
        self._preprocessing = preprocessing
        self._grayscale = grayscale
        if grayscale:
            self._preprocessing.insert(0, vu.RGB_to_grayscale)

    @property
    def movie(self):
        """
            A NumPy array holding the entire movie, loaded only on request. 
            Preferably this will never be used and instead we will use a 
            TensorFlow queue to yield portions of the movie.

            Note there is no way to set the movie.
        """
        if not self._movie_loaded:
            if not self._prealloc:
                self._movie = np.zeros([self.n_frames, self.height, self.width],
                    dtype=np.uint8)
            self._load()
        return self._movie
    @movie.deleter
    def movie(self):
        """
            Deletes movie NumPy array only if it was not preallocated.
        """
        if not self._prealloc and self._movie_loaded:
            del self._movie
            self._movie_loaded = False

    def _load(self):
        """
            A helper function that loads the entire movie.
            Assumes self._movie has already been allocated.
        """
        cap = self._open()
        i_frame = 0
        while cap.isOpened() and i_frame < self.drop_frames:
            ret = cap.grab()
            i_frame += 1
        i_frame -= self.drop_frames
        while cap.isOpened() and i_frame < self.n_frames:
            ret, frame = cap.read()
            processed = reduce(lambda x,f: f(x), self._preprocessing, frame)
            self._movie[i_frame,:,:] = processed
            i_frame += 1
        self._movie_loaded = True
        cap.release()
    def _open(self):
        """
            A probably unnecessary helper function to open
            the movie file and do basic error handling to
            check that it was opened.
        """
        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            raise ValueError("Cannot open movie file path.")
        return cap

    def play(self, window_name='frame'):
        frame_len = int(1000.0 / self.fps)
        for i_frame in xrange(0,self.n_frames):
            cv2.imshow(window_name, np.squeeze(self.movie[i_frame, :, :]))
            if cv2.waitKey(frame_len) & 0xFF == ord('q'):
                break
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
    def save(self, file_name):
        writer = cv2.VideoWriter()
        writer.open(file_name, cv2.VideoWriter_fourcc(*'XVID'),
            self.fps, (self.width, self.height))
        if not writer.isOpened():
            raise Exception("Did not open movie file for writing.")
        for i_frame in xrange(self.n_frames):
            frame = self.movie[i_frame, :, :]
            if self._grayscale:
                frame = vu.grayscale_to_RGB(frame)
            writer.write(frame)
        writer.release()

    def tf_average_luminance(self):
        tf.reduce_mean(self.movie, [1,2])
    def average_luminance(self):
        return np.mean(self.movie, axis=(1,2))
    def luminance_spectrum(self, smooth=None):
        avg_lum = self.average_luminance()
        times = np.arange(0, 1000.0/self.fps, self.n_frames*1000)
        f, Pxx =  signal.periodogram(avg_lum, self.fps)
        if smooth:
            Pxx = smooth(Pxx)
            print 'done smoothing.'
            if len(Pxx) < len(f):
                f = f[-len(Pxx):]
        return f, Pxx


