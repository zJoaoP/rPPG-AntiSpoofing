import numpy as np
import cv2

class Extractor:
	def remove_pixels_outside(self, image, polygon):
		mask = np.zeros(image.shape, dtype = np.uint8)
		cv2.fillPoly(mask, np.array([polygon], dtype = np.int32), color = (1,) * image.shape[2])
		return image * mask

	def remove_pixels_inside(self, image, polygon):
		mask = np.ones(image.shape, dtype = np.uint8)
		cv2.fillPoly(mask, np.array([polygon], dtype = np.int32), color = (0,) * image.shape[2])
		return image * mask

	def extract_points(self, landmarks, index_list):
		points = []
		for i in index_list:
			(x, y) = landmarks[i]
			points += [[x, y]]

		return points

	def extract_roi(self, frame, landmarks):
		return None

class FaceWithoutEyes(Extractor):
	def extract_roi(self, frame, landmarks):
		face_landmark_index_list = list(range(17)) + list(range(26, 17, -1))
		right_eye_index_list = list(range(42, 48))
		left_eye_index_list = list(range(36, 42))
		mouth_index_list = list(range(48, 60))

		face_polygon = self.extract_points(landmarks, face_landmark_index_list)

		right_eye_polygon = self.extract_points(landmarks, right_eye_index_list)
		left_eye_polygon = self.extract_points(landmarks, left_eye_index_list)

		mouth_polygon = self.extract_points(landmarks, mouth_index_list)

		roi = self.remove_pixels_inside(frame, right_eye_polygon)
		roi = self.remove_pixels_inside(roi, left_eye_polygon)
		roi = self.remove_pixels_inside(roi, mouth_polygon)
		roi = self.remove_pixels_outside(roi, face_polygon)
		return roi

class CheeksOnly(Extractor):
	def extract_roi(self, frame, landmarks):
		right_cheek_index_list = [31, 0, 3]
		left_cheek_index_list = [35, 16, 13]

		left_cheek_polygon = self.extract_points(landmarks, left_cheek_index_list)
		right_cheek_polygon = self.extract_points(landmarks, right_cheek_index_list)

		left_cheek_face = self.remove_pixels_outside(frame, left_cheek_polygon)
		right_cheek_face = self.remove_pixels_outside(frame, right_cheek_polygon)

		return left_cheek_face + right_cheek_face

class CheeksAndNose(Extractor):
	def extract_roi(self, frame, landmarks):
		roi_index_list = [0, 3, 31, 35, 13, 16, 28]
		roi_polygon = self.extract_points(landmarks, roi_index_list)
		return self.remove_pixels_outside(frame, roi_polygon)

class LeftCheek(Extractor):
	def extract_roi(self, frame, landmarks):
		roi_index_list = [35, 15, 12]
		roi_polygon = self.extract_points(landmarks, roi_index_list)
		return self.remove_pixels_outside(frame, roi_polygon)

class RightCheek(Extractor):
	def extract_roi(self, frame, landmarks):
		roi_index_list = [31, 1, 4]
		roi_polygon = self.extract_points(landmarks, roi_index_list)
		return self.remove_pixels_outside(frame, roi_polygon)

class Nose(Extractor):
	def extract_roi(self, frame, landmarks):
		roi_index_list = [27, 31, 33, 35]
		roi_polygon = self.extract_points(landmarks, roi_index_list)
		return self.remove_pixels_outside(frame, roi_polygon)