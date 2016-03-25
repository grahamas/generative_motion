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

def dist(x1,x2):
    square_diff = np.power(x1-x2, 2)
    sum_square = np.sum(square_diff)
    return np.sqrt(sum_square)

#session = MatlabSession('matlab -nojvm -nodisplay')
def attach_callback(obj):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            obj.mouse_left_down = True
            obj.circle_center = (x,y)
            obj.circle_radius = 1
        if event == cv2.EVENT_LBUTTONUP:
            obj.mouse_left_down = False
        if event == cv2.EVENT_RBUTTONDOWN and not obj.playing:
            obj.mouse_right = True
        if event == cv2.EVENT_MOUSEMOVE and obj.mouse_left_down:
            obj.circle_radius = max(1, int(dist(np.array([x,y]), obj.circle_center)))
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
        self.video_stream = VideoStream(video_path, grayscale=False)
        self.fps = self.video_stream.fps
        self.circle_center = None
        self.circle_radius = None
        self.mouse_left_down = False
        self.mouse_right = False
        self.default_radius = 10
        self.image = np.zeros((512, 512, 3), dtype=np.uint8)
        self.window_name = 'Flow Tracker'
        self.i_frame = -1
        self.playing = False
        
    def draw_frame(self, frame, frame_len):
        local_frame = frame.copy()
        if self.circle_center is not None:
            print(str(self.circle_center) + ", " + str(self.circle_radius))
            cv2.circle(local_frame, self.circle_center, self.circle_radius, (255, 0, 0), 2)
        cv2.putText(local_frame, str(self.i_frame), (400, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        cv2.imshow(self.window_name, local_frame)
        if cv2.waitKey(frame_len) & 0xFF == ord('q'):
            sys.exit(0)
        return True

    def circle_resultant_vector(self, flow_frame):
        accum = np.array([0,0], dtype=np.float64)
        count = 0
        for i in np.arange(self.circle_center[0]-self.circle_radius,self.circle_center[0]+self.circle_radius):
            for j in np.arange(self.circle_center[1]-self.circle_radius,self.circle_center[1]+self.circle_radius):
                if dist(np.array([i,j]), self.circle_center) <= self.circle_radius:
                    accum += np.squeeze(flow_frame[i,j,:])
                    count += 1
        if count < 1: print "EMPTY CIRCLE"
        return accum / max(1,count)



    def move_circle(self, flow_frame):
        # Recall that (0,0) is in the upper left
        result = self.circle_resultant_vector(flow_frame)
        new_center = map(lambda x: max(int(x),0), np.array(self.circle_center) + result * self.dt)
        self.circle_center = tuple(new_center)

    
    def start_motion_trace(self, video):
        fps = self.fps
        frame_len = int(1000.0 / fps)
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, attach_callback(self))
        for mat_path in self.mat_paths:
            print("FOR!")
            data = np.array(h5py.File(mat_path, 'r').get(self.flow_mat_varname))
            data = data.swapaxes(1,3)
            data = data.swapaxes(1,2)
            # data[:,:,1,:] = data[:,:,1,:] * -1 # Makes vectors have 0,0 as upper left
            frame = video.next()
            self.i_frame += 1
            while not self.mouse_right:
                if not self.draw_frame(frame, frame_len):
                    return
            self.mouse_right = False
            frame = video.next() # Should check for None, but should never be None
            i_flow = 0
            gross_time_start = time.clock()
            self.playing = True
            while frame is not None and i_flow < len(data):
                if not self.mouse_left_down:
                    self.move_circle(data[i_flow,:,:,:])
                    self.draw_frame(frame, 1) # TODO: Get appropriate framelen (based on computation time)
                    i_flow += 1
                    self.i_frame += 1
                    frame = video.next()
                else:
                    self.draw_frame(frame, frame_len)
            self.playing = False
            gross_time = gross_time_start - time.clock()
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.putText(image, 'FPS: ' + str(gross_time / i_flow), (256, 256), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow(self.window_name, image)

if __name__ == '__main__':
    mat_paths = sorted(glob.glob(FLOW_PATH_GLOB), key=zu.split_out_numbers)
    mt = MotionTrace(mat_paths, FLOW_MAT_VARNAME, 
            VIDEO_PATH, OUTPUT_DIR, DT)
    with mt.video_stream as video:
        mt.start_motion_trace(video)
