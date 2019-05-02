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

class Kinect_Wrapper(Wrapper):
	def __init__(self, width = 1920, height = 1080):
		super(Kinect_Wrapper, self).__init__(width, height)

		from pylibfreenect2 import Freenect2, SyncMultiFrameListener, Registration
		from pylibfreenect2 import OpenGLPacketPipeline, FrameType, Frame

		self.pipeline = OpenGLPacketPipeline()
		self.freenect = Freenect2()

		self.num_devices = self.freenect.enumerateDevices()
		assert self.num_devices > 0, "[Kinect Wrapper] Nenhum dispositivo Kinect conectado!"
		
		self.serial = self.freenect.getDeviceSerialNumber(0)
		self.device = self.freenect.openDevice(self.serial, pipeline = self.pipeline)
		self.listener = SyncMultiFrameListener(FrameType.Color)
		
		self.device.setColorFrameListener(self.listener)
		self.device.startStreams(rgb = True, depth = False)

		self.registration = Registration(self.device.getIrCameraParams(), self.device.getColorCameraParams())

	def __del__(self):
		if hasattr(self, "device") and self.device is not None:
			self.device.stop()
			self.device.close()

	def get_frame(self):
		frames = self.listener.waitForNewFrame()
		color_frame = cv2.resize(frames["color"].asarray(np.uint8), (self.width, self.height), cv2.INTER_CUBIC)
		self.listener.release(frames)
		return True, color_frame[:, :, 0 : 3]

	def get_frame_rate(self):
		return 30

class OpenCV_Wrapper(Wrapper):
	def __init__(self, source, width = 1280, height = 720):
		super(OpenCV_Wrapper, self).__init__(width, height)		
		self.stream = cv2.VideoCapture(source)

	def get_frame(self):
		success, frame = self.stream.read()
		if not success:
			return False, frame
			
		return success, frame

	def get_frame_rate(self):
		return int(self.stream.get(cv2.CAP_PROP_FPS))