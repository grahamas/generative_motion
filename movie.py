import os,sys

import numpy as np
from scipy import signal

import cv2
import util.video as vu
from skvideo.io import VideoCapture, VideoWriter

import matplotlib.pyplot as plt

class MovieStream(object):
    """
        Streams a movie from a VideoCapture.
        Does not hold movie; IS ITERATOR.
    """
    def __init__(self, file_path, n_frames=None,
        drop_frames=0, fps=None,
        preprocessing=[], grayscale=True):
        """
            file_path       : path to .avi file
            n_frames        : (optional, defaults to all) number of frames to load
            drop_frames     : (optional) the number of initial frames to drop
            preprocessing   : (default=[]) list of preprocessing methods 
                                (Use grayscale option for grayscale preprocessing)
            grayscale       : (default=True) must specify for loading+saving
        """
        self.file_path = file_path

        # We open the file to get metadata
        cap = self._open()
        # Get the number of movie frames and make sure it is 
        # commensurate with the requested n_frames and drop_frames
        # NOTE: Currently this doesn't work on midway (using old OpenCV)
        # total_n_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        # assert total_n_frames >= n_frames + drop_frames, 
        #     "{} < {} + {}".format(total_n_frames, 
        #     n_frames, drop_frames)
        self._drop_frames = drop_frames
        # Get the movie dimensions
        self.width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        self.fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        if fps > 0:
            if self.fps > 0:
                assert self.fps == fps
            else:
                self.fps = fps
        else:
            assert self.fps > 0
        cap.release()
        if n_frames == 0:
            self.n_frames = sys.maxint - self._drop_frames
        else:
            self.n_frames = n_frames
        self._preprocessing = preprocessing
        self.grayscale = grayscale
        if grayscale:
            self._preprocessing.insert(0,vu.RGB_to_grayscale)

    # Methods for iteration
    def __iter__(self):
        """
            Initializes self as iterator by skipping dropped frames.
        """
        self._cap = self._open()
        self.i_frame = 0
        while self.i_frame < self._drop_frames:
            ret = self._cap.grab()
            self.i_frame += 1
        return self
    def next(self):
        """
            Iterates while cap remains opened and we haven't advanced beyond n_frames.
            Applies pre-processing at every call. 
        """
        if self._cap.isOpened() and self.i_frame < self.n_frames + self._drop_frames:
            ret, frame = self._cap.read()
            if frame is None:
                raise StopIteration
            processed = reduce(lambda x,f: f(x), self._preprocessing, frame)
            self.i_frame += 1
            return processed
        else:
            raise StopIteration

    def _open(self):
        """
            A probably unnecessary helper function to open
            the movie file and do basic error handling to
            check that it was opened.
        """
        cap = cv2.VideoCapture(os.path.realpath(self.file_path))
        if not cap.isOpened():
            raise ValueError("Cannot open movie file path.")
        return cap

    def play(self, window_name='frame'):
        """
            Play the movie using OpenCV.
        """
        frame_len = int(1000.0 / self.fps)
        for frame in self:
            print frame
            cv2.imshow(window_name, np.array(np.squeeze(frame), dtype=np.uint8))
            if cv2.waitKey(frame_len) & 0xFF == ord('q'):
                break
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)

    def save(self, file_name):
        """
            Save the movie to file_name as XVID avi
        """
        writer = cv2.VideoWriter()
        writer.open(file_name, cv2.cv.CV_FOURCC(*'XVID'),
            self.fps, (self.width, self.height))
        if not writer.isOpened():
            raise Exception("Did not open movie file for writing.")
        for frame in self:
            if self.grayscale:
                frame = vu.grayscale_to_RGB(frame)
            writer.write(frame)
        writer.release()
    
    # TODO
    def save_tfrecord(self, name, path, num_records=1):
        """
            Save the movie as a TFRecord.
        """
        if num_records is not 1:
            raise NotImplementedError('Does not yet support splitting records.')
        filename = os.path.join(path, name, '.tfrecord')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for frame in self:
            frame = vu.binary(frame).tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'frame_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

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

