import cv2

class Wrapper:
	def __init__(self, width = 1280, height = 720):
		self.height = height
		self.width = width

	def get_frame(self): #Not Success, None
		return False, None

	def get_frame_rate(self):
		return 0

	def stop_condition(self):
		return True

class OpenCV_Wrapper(Wrapper):
	def __init__(self, source, width = 1280, height = 720):
		super(OpenCV_Wrapper, self).__init__(width, height)		
		self.stream = cv2.VideoCapture(source)

	def get_frame(self):
		return self.stream.read()

	def get_frame_rate(self):
		return int(self.stream.get(cv2.CAP_PROP_FPS))

	def stop_condition(self):
		return cv2.waitKey(1) & 0xFF == ord('q')