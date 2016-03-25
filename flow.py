import os, sys, time

import numpy as np
import h5py

import cv2

#from pymatlab.matlab import MatlabSession

import glob
import util.stringz as zu
from data.stream import Stream
from data.video import VideoStream

VIDEO_PATH = '../movies/glassframe1.avi'
DT=1

SOURCE_DIR = "../movies/flow/"
OUTPUT_DIR = "/project/sepalmer/grahams/flow/"

FLOW_PATH_GLOB = SOURCE_DIR + 'glassframe1*.mat'
FLOW_MAT_VARNAME = 'uv_mov'


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
    # Assumes shape of matrix, easy fix
    def __init__(self, mat_paths, mat_varname, video_path, output_path,
            dt=1):
        self.video_path = video_path
        self.mat_paths = mat_paths
        self.flow_mat_varname = mat_varname
        self.output_path = output_path
        self.dt = dt
        print("going to video")
        self.video_stream = VideoStream(video_path, grayscale=False)
        print("out!")
        self.fps = self.video_stream.fps
        self.circle_center = None
        self.circle_radius = None
        self.mouse_left_down = False
        self.mouse_right = False
        self.default_radius = 10
        self.image = np.zeros((512, 512, 3), dtype=np.uint8)
        self.window_name = 'Flow Tracker'
        self.i_frame = -1
        
    def draw_frame(self, frame):
        if self.circle_center is not None:
            cv2.circle(frame, self.circle_center, self.circle_radius, (255, 0, 0), -1)
        cv2.putText(frame, str(i_frame), (450, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(frame_len) & 0xFF == ord('q'):
            sys.exit(0)
        return True

    def circle_resultant_vector(self, flow_frame):
        accum = np.array([0,0])
        count = 0
        for i in np.arange(512):
            for j in np.arange(512):
                if np.sqrt(np.sum(np.array([i,512-j]) - self.circle_center)) <= self.circle_radius:
                    accum += flow_frame[i,j]
                    count += 1
        return accum / count



    def move_circle(self, flow_frame):
        # Recall that (0,0) is in the upper left
        result = self.circle_resultant_vector(flow_frame)
        self.circle_center += result * self.dt

    
    def start_motion_trace(self, video):
        fps = self.fps
        frame_len = (1000.0 / fps)
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, attach_callback(self))
        print("for?")
        print self.mat_paths
        for mat_path in self.mat_paths:
            print("FOR!")
            data = np.array(h5py.File(mat_path, 'r').get(self.flow_mat_varname))
            # data[:,:,1,:] = data[:,:,1,:] * -1 # Makes vectors have 0,0 as upper left
            print('next!')
            frame = np.squeeze(video.next())
            print('after next...')
            self.i_frame += 1
            while not mouse_right:
                if not draw_frame(frame, frame_len):
                    return
            frame = np.squeeze(video.next())
            i_flow = 0
            gross_time_start = time.clock()
            while frame is not None:
                self.move_circle(data[:,:,:,i_flow])
                self.draw_frame(frame, frame_len)
            gross_time = gross_time_start - time.clock()
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.putText(image, 'FPS: ' + str(gross_time / i_flow), (256, 256), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow(self.window_name, image)

if __name__ == '__main__':
    mat_paths = sorted(glob.glob(FLOW_PATH_GLOB), key=zu.split_out_numbers)
    mt = MotionTrace(mat_paths, FLOW_MAT_VARNAME, 
            VIDEO_PATH, OUTPUT_DIR, DT)
    print("with?")
    with mt.video_stream as video:
        print("with!")
        mt.start_motion_trace(video)
        print("yay!")
