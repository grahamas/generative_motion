import os, sys

import numpy as np
import h5py

import cv2

#from pymatlab.matlab import MatlabSession

import glob
import util.stringz as zu
from data.stream import Stream
from data.video import VideoStream

video_path = '../movies/glassframe1.avi'

flow_source_dir = "../movies/flow/"
flow_target_dir = "/project/sepalmer/grahams/flow/"

flow_path_glob = flow_source_dir + 'glassframe1*.mat'
flow_mat_varname = 'uv_mov'

flow_mats = sorted(glob.glob(flow_path_glob), key=zu.split_out_numbers)

#session = MatlabSession('matlab -nojvm -nodisplay')
def attach_callback(obj):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            obj.mouse_left_down = True
            obj.circle_center = (x,y)
            obj.circle_radius = obj.default_radius
        if event == cv2.EVENT_RBUTTONDOWN:
            obj.mouse_right = True
        if event == cv2.EVENT_MOUSEMOVE and objmouse_left_down:
            pass
    return mouse_callback

class MotionTrace(object):

    def __init__(self, video_path, mat_paths, target_path):
        self.video_path = video_path
        self.mat_paths = mat_paths
        self.target_path = target_path
        self.video_stream = VideoStream(video_path, grayscale=False)
        self.circle_center = None
        self.circle_radius = None
        self.mouse_left_down = False
        self.mouse_right = False
        self.default_radius = 20
        self.image = np.zeros((512, 512, 3), dtype=np.uint8)
        self.window_name = 'Flow Tracker'
        self.i_frame = -1
        
    def draw_frame(self, img, window_name, i_frame):
        if self.circle_center is not None:
            cv2.circle(img, self.circle_center, self.circle_radius, (255, 0, 0), -1)
        cv2.putText(img, str(i_frame), (450, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow(window_name, img)
        if cv2.waitKey(frame_len) & 0xFF == ord('q'):
            sys.exit(0)
        return True
    
    def start_motion_trace(self, video):
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name,mouse_callback)
        for mat_path in self.mat_paths:
            data = np.array(h5py.File(mat_path, 'r').get(flow_mat_varname))
            frame = np.squeeze(video.next())
            self.i_frame += 1
            while not mouse_right:
                image = frame
                if not draw_frame(image, window_name):
                    return
            while frame is not None:
                pass

        


with video_stream as video:
    motion_trace(flow_mats, video)
