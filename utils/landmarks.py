import numpy as np
import dlib
import cv2

class LandmarkPredictor:
	def __init__(self, predictor_folder = "./shape_predictor/landmark_predictor.dat", detection_window = 120):
		self.predictor = dlib.shape_predictor(predictor_folder)
		self.detector = dlib.get_frontal_face_detector()
		self.detection_window = detection_window
		self.current_frame = 0
		self.tracker = None

	def rect_to_bbox(self, rect):
		x = rect.left()
		y = rect.top()
		w = rect.right() - x
		h = rect.bottom() - y
		return (x, y, w, h)

	def drectangle_to_rectangle(self, drect):
		x = int(drect.left())
		y = int(drect.top())
		p = int(drect.right())
		q = int(drect.bottom())
		return dlib.rectangle(x, y, p, q)

	def to_gray(self, image):
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	def shape_to_np(self, shape):
		coords = np.zeros([68, 2], dtype = "int")
		
		for i in range(68):
			coords[i] = (shape.part(i).x, shape.part(i).y)

		return coords

	def detect_landmarks(self, image, rect):
		gray = self.to_gray(image)

		prediction = self.predictor(gray, rect)
		return self.shape_to_np(prediction)

	def detect_face(self, image):
		gray = self.to_gray(image)
		if self.tracker is None:
			print("[Detector] Trying to initialize tracker.")
			face_rect = self.detector(gray, 1)
			if len(face_rect) == 0:
				return None
			else:
				face_rect = face_rect[0]
			print("[Detector] Face found. Initializing tracker!")
			
			self.tracker = dlib.correlation_tracker()
			self.tracker.start_track(image, face_rect)
		elif (self.current_frame == self.detection_window) and (self.tracker is not None):
			print("[Detector] Trying to update the tracker region.")
			
			face_rect = self.detector(gray, 1)
			if len(face_rect) == 0:
				return None
			else:
				face_rect = face_rect[0]

			self.tracker.start_track(image, face_rect)
			self.current_frame = 0

		self.current_frame += 1
		self.tracker.update(image)
		return self.drectangle_to_rectangle(self.tracker.get_position())