from utils.extractor import LeftCheek, RightCheek, Nose, CheeksAndNose
from utils.landmarks import LandmarkPredictor
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


class OpenCV_Wrapper:
	def __init__(self, source):
		self.stream = cv2.VideoCapture(source)

	def read(self):
		success, frame = self.stream.read()
		if not success:
			return False, None
		else:
			return success, frame

	def lenght(self):
		frame_count = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
		return frame_count if frame_count > 0 else (1 << 27)


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

		left_cheek, right_cheek, nose, cheeks_and_nose = LeftCheek(), RightCheek(), Nose(), CheeksAndNose()
		loader = OpenCV_Wrapper(source=source)
		predictor = LandmarkPredictor()

		if annotation is not None:
			annotations = load_annotations(annotation)
		
		cheeks_and_nose_features = np.empty([loader.lenght(), 3], dtype=np.float32)
		right_cheek_features = np.empty([loader.lenght(), 3], dtype=np.float32)
		left_cheek_features = np.empty([loader.lenght(), 3], dtype=np.float32)
		nose_features = np.empty([loader.lenght(), 3], dtype=np.float32)
		for i in range(loader.lenght()):
			success, frame = loader.read()
			cv2.waitKey(1) & 0xFF

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

			def get_roi_mean(roi):
				roi[roi < 1.0] = np.nan
				r, g, b = np.nanmean(roi, axis=(0, 1))
				for c in [r, g, b]:
					if np.isnan(c):
						c = 0
					
				return np.array([[r, g, b]], dtype=np.float32)

			if (annotation is not None) and not valid_annotation(annotations, i):
				if i == 0:
					cheeks_and_nose_features[i] = 0.0
					right_cheek_features[i] = 0.0
					left_cheek_features[i] = 0.0
					nose_features[i] = 0.0
				else:
					cheeks_and_nose_features[i] = cheeks_and_nose_features[i - 1]
					right_cheek_features[i] = right_cheek_features[i - 1]
					left_cheek_features[i] = left_cheek_features[i - 1]
					nose_features[i] = nose_features[i - 1]
			else:
				# Corrigir os casos sem anotações.
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
				
				cheeks_and_nose_roi = cheeks_and_nose.extract_roi(frame, landmarks).astype(np.float32)
				right_cheek_roi = right_cheek.extract_roi(frame, landmarks).astype(np.float32)
				left_cheek_roi = left_cheek.extract_roi(frame, landmarks).astype(np.float32)
				nose_roi = nose.extract_roi(frame, landmarks).astype(np.float32)

				cheeks_and_nose_features[i] = get_roi_mean(cheeks_and_nose_roi)
				right_cheek_features[i] = get_roi_mean(right_cheek_roi)
				left_cheek_features[i] = get_roi_mean(left_cheek_roi)
				nose_features[i] = get_roi_mean(nose_roi)

		print("[DataLoader] Features extracted from '{0}'.".format(source))

		# import matplotlib.pyplot as plt

		# old_data = np.load('./data/old/rad_train_real.npy')
		# for i, c in enumerate('rgb'):
		# 	plt.plot(old_data[0][:, i], '{}--'.format(c))

		# for i, c in enumerate('rgb'):
		# 	plt.plot(cheeks_and_nose_features[:, i], c)

		# plt.show()

		# raise Exception("UAHEUEHUEAH")

		global_features = np.append(left_cheek_features, nose_features, axis=1)
		global_features = np.append(global_features, right_cheek_features, axis=1)
		return np.append(global_features, cheeks_and_nose_features, axis=1)

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
											video_data.shape[1]], dtype=np.float32)

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


from algorithms.de_haan import DeHaan
from algorithms.wang import Wang
from algorithms.pbv import PBV
from copy import deepcopy

def get_rppg_data(data, frame_rate=24):
	data = deepcopy(data)

	final_data = None
	for c in range(data.shape[-1] // 3):
		roi_data = data[:, :, c*3:(c+1)*3]
		nan_locations = np.isnan(roi_data)
		roi_data[nan_locations] = 0.0

		current_data = np.empty([roi_data.shape[0], roi_data.shape[1], 1])
		for i in range(len(roi_data)):
			current_rppg = Wang.extract_rppg(roi_data[i], frame_rate)
			# _, current_fft = DeHaan.get_fft(current_rppg, frame_rate)

			current_data[i] = current_rppg.reshape(current_rppg.shape[0], 1)
			# current_data[i] = current_fft.reshape(current_fft.shape[0], 1)

			if i % 128 == 0:
				print("[{} / {}] Extracting ppg from roi id = {}.".format(i, len(current_data), c))

		if final_data is None:
			final_data = current_data
		else:
			final_data = np.append(final_data, current_data, axis=-1)

	return final_data
