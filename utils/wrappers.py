from pylibfreenect2 import Freenect2, SyncMultiFrameListener, Registration
from pylibfreenect2 import OpenGLPacketPipeline, FrameType, Frame

import numpy as np
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

class Kinect_Wrapper(Wrapper):

	def __init__(self, width = 680, height = 440):
		super(Kinect_Wrapper, self).__init__(width, height)

		self.pipeline = OpenGLPacketPipeline()
		self.freenect = Freenect2()
		
		self.serial = self.freenect.getDeviceSerialNumber(0)
		self.device = self.freenect.openDevice(self.serial, pipeline = self.pipeline)
		self.listener = SyncMultiFrameListener(FrameType.Color)
		
		self.device.setColorFrameListener(self.listener)
		self.device.start()

		self.registration = Registration(self.device.getIrCameraParams(), self.device.getColorCameraParams())

	def __del__(self):
		self.device.stop()
		self.device.close()

	def get_frame(self):
		frames = self.listener.waitForNewFrame()
		rgb_frame = cv2.resize(frames["color"].asarray(), (self.width, self.height))
		self.listener.release(frames)
		return True, cv2.cvtColor(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)

	def get_frame_rate(self):
		return 30

class OpenCV_Wrapper(Wrapper):
	def __init__(self, source, width = 1280, height = 720):
		super(OpenCV_Wrapper, self).__init__(width, height)		
		self.stream = cv2.VideoCapture(source)

	def get_frame(self):
		return self.stream.read()

	def get_frame_rate(self):
		return int(self.stream.get(cv2.CAP_PROP_FPS))