import os

import numpy as np
import tensorflow as tf

import cv2
import video_utils as vu

class Movie():
	"""Holds a movie."""
	def __init__(self, file_path, storage_array=None,
		n_frames=0, drop_frames=0, buffer=None,
		binarize=False):
		"""
		TODO: Integrate this with Tensorflow queueing. Meaning, don't
				load the entire thing into memory.
			file_path 		: absolute or relative path to .avi file
			storage_array	: (optional) preallocated storage
			n_frames 		: (optional) the number of frames to take
			drop_frames		: (optional) the number of initial frames to drop
			binarize		: (default=False) apply threshold to make movie b/w
		"""
		self.file_path = file_path

		# We open the file to get metadata
		cap = self._open()
		# Get the number of movie frames and make sure it is 
		# commensurate with the requested n_frames and drop_frames
		total_n_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
		assert total_n_frames >= n_frames + drop_frames
		if n_frames == 0: 
			self.n_frames = total_n_frames
		else:
			assert n_frames > 0
			self.n_frames = n_frames
		# Get the movie dimensions
		self.width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
		self.height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
		self.fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
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
	    		self._movie = np.zeros([self.n_frames, self.height, self.width])
	    	self._load()
	    return self._movie
	@movie.deleter
	def movie(self):
		"""
			Deletes movie NumPy array only if it was not preallocated.
		"""
		if not self._prealloc and self._movie:
			del self._movie

	def _load(self):
		"""
			A helper function that loads the entire movie.
			Assumes self._movie has already been allocated.
		"""
		cap = self._open()
		i_frame = 0
		while cap.isOpened() and i_frame < self.drop_frames:
			ret, frame = cap.read()
			i_frame += 1
		i_frame -= self.drop_frames
		while cap.isOpened():
			ret, frame = cap.read()
			self._movie[i_frame,:,:] = frame
	def _open(self):
		"""
			A probably unnecessary helper function to open
			the movie file and do basic error handling to
			check that it was opened.
		"""
		cap = cv2.VideoCapture(self.file_path)
		if not cap.isOpened():
			raise ValueError("Cannot open movie file path.")
