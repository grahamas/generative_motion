import movie as movie
import util.video as vu
import cv2
import json
import os

with open('../movies.json') as data_file:    
        struct = json.load(data_file)
        movies_path = struct['movies_path']
        l_movie_args = struct['movie_args']

i_movie = 0
movie_args = l_movie_args[i_movie]
print movies_path, movie_args['fn']
print os.path.isfile(os.path.join(movies_path,movie_args['fn']))
mov = movie.MovieStream(os.path.join(movies_path,movie_args['fn']).encode('ascii','ignore'),
        drop_frames=movie_args['drop_frames'],
        n_frames=1000, fps=60, preprocessing=[vu.binarize_otsu])
mov.save_tfrecord('iamsaved')
