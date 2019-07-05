import numpy as np
import threading
import argparse
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
		print("[DataLoader] Extracting features from '{0}'.".format(source))

		def load_annotations(annotation_location):
			with open(annotation_location) as file:
				annotations = []
				for line in file.readlines():
					annotation_line = [int(i) for i in line[:-1].split(' ') if i.isdigit()]
					if len(annotation_line) > 4:
						annotation_line = annotation_line[-4:]

					annotations += [annotation_line]

			return np.array(annotations)

		loader = AsyncVideoLoader(source=source).start()
		landmark_predictor = LandmarkPredictor()
		extractor = CheeksAndNose()

		features = np.empty([loader.lenght(), 3], dtype=np.float32)
		annotations = load_annotations(annotation)

		for i in range(loader.lenght()):
			success, frame = loader.read()
			cv2.waitKey(1) & 0xFF

			if (i >= len(annotations)) or (np.sum(annotations[i]) == 0):
				features[i] = features[i - 1]
			else:
				x, y, w, h = annotations[i]
				frame = frame[y:y+h, x:x+w]

				landmarks = landmark_predictor.detect_landmarks(frame)
				frame = extractor.extract_roi(frame, landmarks).astype(np.float32)

				frame[frame < 1.0] = np.nan
				features[i] = np.nanmean(frame, axis=(0, 1))

		loader.stop()
		
		print("[DataLoader] Features extracted from '{0}'.".format(source))
		return features

class GenericDatasetLoader:
	@staticmethod
	def walk_and_load_from(root_folder, annotations_folder):
		def is_video(filename):
			return 	filename.endswith(".mp4")\
					or filename.endswith(".mov")

		def is_annotation(filename):
			return 	filename.endswith(".face")

		video_locations = list()
		for root, dirs, files in os.walk(root_folder):
			for file in files:
				if is_video(file):
					video_locations.append("{0}/{1}".format(root, file))

		annotations = list()
		for root, dirs, files in os.walk(annotations_folder):
			for file in files:
				if is_annotation(file):
					annotations.append("{0}/{1}".format(root, file))

		video_locations = sorted(video_locations)
		annotations = sorted(annotations)
		folder_features = None
		k = 0
		for video, annotation in zip(video_locations, annotations):
			video_data = AnnotatedVideoLoader.load_features(
							source=video,
							annotation=annotation)

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


class ReplayAttackLoader:
	@staticmethod
	def __load_by_split(source, split):
		split_real = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/real".format(source, split),
			"{0}/face-locations/{1}/real".format(source, split)
		)

		split_attack = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/attack".format(source, split),
			"{0}/face-locations/{1}/attack".format(source, split)
		)

		return split_attack, split_real

	@staticmethod
	def load_train(source):
		return ReplayAttackLoader.__load_by_split(source, 'train')

	@staticmethod
	def load_devel(source):
		return ReplayAttackLoader.__load_by_split(source, 'devel')

	@staticmethod
	def load_test(source):
		return ReplayAttackLoader.__load_by_split(source, 'test')

	@staticmethod
	def load_and_store(source, destination):
		train_attack, train_real = ReplayAttackLoader.load_train(source)
		devel_attack, devel_real = ReplayAttackLoader.load_devel(source)
		test_attack, test_real = ReplayAttackLoader.load_test(source)

		np.save("{0}/rad_train_fake.npy".format(destination), train_attack)
		np.save("{0}/rad_train_real.npy".format(destination), train_real)

		np.save("{0}/rad_test_fake.npy".format(destination), test_attack)
		np.save("{0}/rad_test_real.npy".format(destination), test_real)

		np.save("{0}/rad_devel_fake.npy".format(destination), devel_attack)
		np.save("{0}/rad_devel_real.npy".format(destination), devel_real)

		pass


class SpoofInTheWildLoader:
	@staticmethod
	def __load_by_split(source, split):
		split_live = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/live".format(source, split),
			"{0}/{1}/live".format(source, split)
		)

		split_spoof = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/spoof".format(source, split),
			"{0}/{1}/spoof".format(source, split)
		)

		return split_live, split_spoof

	@staticmethod
	def load_train(source):
		return SpoofInTheWildLoader.__load_by_split(source, 'Train')

	@staticmethod
	def load_test(source):
		return SpoofInTheWildLoader.__load_by_split(source, 'Test')

	@staticmethod
	def load_and_store(source, destination):
		train_live, train_spoof = SpoofInTheWildLoader.load_train(source)
		test_live, test_spoof = SpoofInTheWildLoader.load_test(source)

		np.save("{0}/siw_train_live.npy", train_live)
		np.save("{0}/siw_train_spoof.npy", train_spoof)

		np.save("{0}/siw_test_live.npy", test_live)
		np.save("{0}/siw_test_spoof.npy", test_spoof)

		pass


def get_args():
	parser = argparse.ArgumentParser(description="Código para extração das \
										bases de dados de PAD como SiW e \
										ReplayAttack")

	parser.add_argument("source", nargs=None, default=None, action="store")
	parser.add_argument("dest", nargs=None, default=None, action="store")
	parser.add_argument("time", nargs=None, default=5, action="store", type=int)

	return parser.parse_args()

if __name__ == "__main__":
	args = get_args()
	if args.source.endswith('ReplayAttack'):
		ReplayAttackLoader.load_and_store(args.source, args.dest)
	elif args.source.endswith('SiW_release'):
		SpoofInTheWildLoader.load_and_store(args.source, args.dest)
	else:
		print("Base de dados não suportada.")