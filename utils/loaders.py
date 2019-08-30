from utils.landmarks import LandmarkPredictor
from utils.extractor import CheeksAndNose
import numpy as np
import threading
import dlib
import cv2
import os


class AsyncVideoLoader:
	"""
	This class can load videos using parallel processing and threads.
	
	Attributes:
		source (string): Target video location.
	"""
	def __init__(self, source):
		self.capture = cv2.VideoCapture(source)

		self.read_lock = threading.Lock()
		self.started = False

		self.success, self.frame = self.capture.read() 

	# This method starts the class thread, unless it already started.
	def start(self):
		if self.started is True:
			pass

		self.thread = threading.Thread(target=self.update, args=())
		self.started = True
		self.thread.start()

		return self

	def stop(self):
		self.started = False

	#This method update the current frame received from buffer.
	def update(self):
		while self.started:
			success, frame = self.capture.read()
			if not success:
				self.stop()
			else:
				with self.read_lock:
					self.success = success
					self.frame = frame

	#This method return the current frame captured from buffer. (Thread safe?)
	def read(self):
		with self.read_lock:
			frame = self.frame.copy()
			success = self.success

		return success, frame

	def lenght(self):
		return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

	def __exit__(self, exec_type, exc_value, traceback):
		self.capture.release()


"""
Class dedicated to load videos and crop from desired location \
using the AsyncLoader defined previously and cropping the region of interest.
"""
class VideoLoader:
	"""
	Method name: load_features

	This method represents the main method of this class. It loads a video \
	based on source string and crop according to the information present in the
	annotation files or dettect faces with the "dlib" library.


	Attributes:
		source (string): The source location of desired video.
		annotation (string): The source location of desired video annotation.
		border_notation (bool): Is related to the current data format of \
								annotation file. \
								If True, [x, y, u, v] describes (x, y) as \
								the left supperior points and (u, v) the right \
								inferior points.
								If False, it means [x, y, width, height].
	"""
	@staticmethod
	def load_features(source, annotation=None, border_notation=False):
		print("[DataLoader] Extracting features from '{0}'.".format(source))

		def load_annotations(annotation_location):
			def is_digit(n):
				try:
					int(n)
					return True
				except ValueError:
					return False

			with open(annotation_location) as file:
				annotations = []
				for line in file.readlines():
					line = line.replace(',', ' ')
					annotation_line = list()
					for i in line[:-1].split(' '):
						if is_digit(i):
							annotation_line.append(int(i))

					if len(annotation_line) > 4:
						annotation_line = annotation_line[-4:]

					annotations += [annotation_line]

			return np.array(annotations)

		loader = AsyncVideoLoader(source=source).start()
		predictor = LandmarkPredictor()
		extractor = CheeksAndNose()
		if annotation is not None:
			annotations = load_annotations(annotation)
		
		features = np.zeros([loader.lenght(), 3], dtype=np.float32)
		for i in range(loader.lenght()):
			success, frame = loader.read()
			def valid_annotation(annotations, current_index):
				if len(annotations) < current_index:
					return False
				elif len(annotations[current_index]) != 4:
					return False
				else:
					for i in annotations[current_index]:
						if i <= 0:
							return False
					return True

			if (annotation is not None) and not valid_annotation(annotations, i):
				features[i] = 0 if (i == 0) else features[i - 1]
			else:
				if annotation is None:
					rect = predictor.detect_face(frame)
					if rect is None:
						features[i] = 0 if (i == 0) else features[i - 1]
					else:
						left = rect.left()
						right = rect.right()
						top = rect.top()
						bottom = rect.bottom()

						frame = frame[top:bottom, left:right]
				else:				
					x, y, w, h = annotations[i]						
					if not border_notation:
						frame = frame[y:y+h, x:x+w]
					else:
						frame = frame[y:h, x:w]

				landmarks = predictor.detect_landmarks(frame)
				frame = extractor.extract_roi(frame, landmarks).astype(np.float32)

				frame[frame == 0.0] = np.nan
				features[i] = np.nanmean(frame, axis=(0, 1))
				if np.any(np.isnan(features[i])):
					features[i] = features[i - 1] if i > 0 else 0.0

		loader.stop()
		print("[DataLoader] Features extracted from '{0}'.".format(source))
		return features

class GenericDatasetLoader:
	@staticmethod
	def walk_and_load_from(root_folder, annotations_folder=None, border_notation=False):
		def is_video(filename):
			return 	filename.endswith(".mp4")\
					or filename.endswith(".mov")\
					or filename.endswith(".avi")

		def is_annotation(filename):
			return 	filename.endswith(".face")\
					or filename.endswith(".txt")

		video_locations = list()
		for root, dirs, files in os.walk(root_folder):
			for file in files:
				if is_video(file):
					video_locations.append("{0}/{1}".format(root, file))

		if annotations_folder is not None:
			annotations = list()
			for root, dirs, files in os.walk(annotations_folder):
				for file in files:
					if is_annotation(file):
						annotations.append("{0}/{1}".format(root, file))
	
			annotations = sorted(annotations)
		else:
			annotations = None

		video_locations = sorted(video_locations)
		folder_features = None
		k = 0
		for i in range(len(video_locations)):
			video = video_locations[i]
			annotation = annotations[i] if annotations is not None else None

			video_data = VideoLoader.load_features(
							source=video,
							annotation=annotation,
							border_notation=border_notation)

			if folder_features is None:
				folder_features = np.empty([len(video_locations),
											video_data.shape[0],
											3], dtype=np.float32)

			else:
				if video_data.shape[0] < folder_features.shape[1]:
					new_video_data = np.zeros(folder_features.shape[1:])
					new_video_data[:video_data.shape[0]] = video_data
					video_data = new_video_data.copy()

				elif video_data.shape[0] > folder_features.shape[1]:
					video_data = video_data[:folder_features.shape[1]]

			folder_features[k] = video_data			
			k += 1
		
		return folder_features

def slice_and_stride(data, size, stride):	
	def slice_and_stride_video(video):
		slices = list()
		for i in range(0, len(video), stride):
			if i + size <= len(video):
				slices.append(video[i:i+size])
		
		return np.array(slices)

	data_slice = None
	for i in range(len(data)):
		sl = slice_and_stride_video(data[i])
		if data_slice is None:
			data_slice = sl
		else:
			data_slice = np.append(data_slice, sl, axis=0)

	return data_slice

from algorithms.wang import Wang
def get_rppg_data(data, frame_rate=25):
	final_data = None
	nan_locations = np.isnan(data)
	data[nan_locations] = 0.0
	for i in range(len(data)):
		current_rppg = Wang.extract_rppg(data[i], frame_rate)
		if final_data is None:
			final_data = np.empty([len(data), current_rppg.shape[0], 1])

		final_data[i] = current_rppg.reshape(current_rppg.shape[0], 1)

		if (i > 0) and (i % 256 == 0):
			print("[Wang] {0} / {1} rPPG features processed.".format(i,
																	 len(data)))

	print(final_data.shape)
	return final_data
	