import cv2

def play_video(vc, window_name='frame', fps=60):
	frame_len = int((1.0 / fps) * 1000.0)
	while (vc.isOpened()):
		ret, frame = vc.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow(window_name,gray)
		if cv2.waitKey(frame_len) & 0xFF == ord('q'):
			break
