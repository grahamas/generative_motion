import cv2
import numpy as np

def play_video_cap(vc, window_name='frame', fps=60, processing=[]):
    frame_len = int(1000.0 / fps)
    while (vc.isOpened()):
        ret, frame = vc.read()
        print 'Frame: ',frame
        processed = reduce(lambda x,f: f(x), processing, frame)
        cv2.imshow(window_name, processed)
        if cv2.waitKey(frame_len) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

def play_video_array(arr, fps=60):
    frame_len = int(1000.0/fps)
    shape = np.shape(arr)

def binarize_median(frame):
    median = np.median(frame)
    ret,th = cv2.threshold(frame,median,255,cv2.THRESH_BINARY)
    return th
def binarize_otsu(frame):
    frame = np.array(frame, np.uint8)
    ret,th = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th
def binarize_gotsu(frame):
    blur = cv2.GaussianBlur(frame,(5,5),0)
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def RGB_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
def grayscale_to_RGB(frame):
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

def frame_byte_string(frame):
    # Note that TensorFlow supports maximum unsigned integer
    # bit length of 16, which is what drove this choice
    EXPECTED_SIZE = (512, 512)
    INT_LENGTH = 16
    b_frame = np.array(frame > 0, dtype=uint16)
    # TODO: Generalize assertion
    size = np.size(frame)
    assert size == EXPECTED_SIZE
    height, width = size
    assert np.mod(width, INT_LENGTH) == 0
    cols = np.divide(width, INT_LENGTH)
    pows = array([32768, 16384,  8192,  4096,  2048,  1024,   512,   256,   128,
              64,    32,    16,     8,     4,     2,     1], dtype=uint16)
    print "HELLO"
    #out = np.zeros((height, cols), dtype=uint16)
    split = np.hsplit(frame, cols)
    out = np.linalg.matmul(split, pows)#, out)
    return out.tostring()

