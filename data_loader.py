import numpy as np
import threading
import time
import cv2
import os


from utils.landmarks import LandmarkPredictor
from utils.extractor import CheeksAndNose


class AsyncVideoLoader:
	def __init__(self, source):
		self.capture = cv2.VideoCapture(source)
		self.read_lock = threading.Lock()
		self.started = False

		self.success, self.frame = self.capture.read() 

	def start(self):
		if self.started is True:
			pass

		self.thread = threading.Thread(target=self.update, args=())
		self.started = True
		self.thread.start()

		return self

	def stop(self):
		self.started = False

	def update(self):
		while self.started:
			success, frame = self.capture.read()
			if not success:
				self.stop()
			else:
				with self.read_lock:
					self.success = success
					self.frame = frame

	def read(self):
		with self.read_lock:
			frame = self.frame.copy()
			success = self.success

		return success, frame

	def lenght(self):
		return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

	def __exit__(self, exec_type, exc_value, traceback):
		self.capture.release()

class AnnotatedVideoLoader:
	@staticmethod
	def load_features(source, annotation):
		def load_annotations(annotation_location):
			pass

		loader = AsyncVideoLoader(source=source).start()
		landmark_predictor = LandmarkPredictor()
		extractor = CheeksAndNose()

		annotations = load_annotations(annotation_folder)

		features = np.empty([loader.lenght(), 3], dtype=np.float32)
		for i in range(loader.lenght()):
			success, frame = loader.read()
			cv2.waitKey(1) & 0xFF

			# annotations here.
			frame = frame[250:500, 250:500, :]

			landmarks = landmark_predictor.detect_landmarks(frame)

			frame = extractor.extract_roi(frame, landmarks).astype(np.float32)

			# cv2.imshow("CutFrame", frame.astype(np.uint8))

			frame[frame < 1.0] = np.nan
			features[i] = np.nanmean(frame, axis=(0, 1))

		loader.stop()
		return features

class GenericDatasetLoader:
	@staticmethod
	def walk_and_load_from(root_folder, annotations_folder):
		def is_video(filename):
			return 	filename.endswith(".mp4")\
					or filename.endswith(".avi")

		def is_annotation(filename):
			return 	filename.endswith(".mp4")\
					or filename.endswith(".avi")			

		video_locations = list()
		for root, dirs, files in os.walk(root_folder):
			for file in files:
				if is_video(file):
					video_locations.append("{0}/{1}".format(root, file))

		annotations = list()
		for root, dirs, files in os.walk(annotations_folder):
			for file in files:
				if is_video(file):
					annotations.append("{0}/{1}".format(root, file))

		for i in range(len(video_locations)):
			video_data = AnnotatedVideoLoader.load_features(
							source=video_locations[i],
							annotation=annotations[i])
			
			print(video_data.shape)
		
		print(sorted(video_locations))
		print(sorted(annotations))

if __name__ == "__main__":
	GenericDatasetLoader.walk_and_load_from(
		"./videos/ReplayAttack/train/fake",
		"./videos/ReplayAttack/annotations/train/fake"
	)

	# t0 = time.time()
	# features = AnnotatedVideoLoader.load_features("./videos/motion.avi", None)
	# final_time = time.time() - t0
	# print(features.shape)
	# print(900.0 / final_time)