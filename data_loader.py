import numpy as np
import threading
import argparse
import time
import cv2
import os


from utils.landmarks import LandmarkPredictor
from utils.extractor import CheeksAndNose


class AsyncVideoLoader:
	def __init__(self, source, width=None, height=None):
		self.capture = cv2.VideoCapture(source)
		if (width is not None) and (height is not None):
			self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width);
			self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height);

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
	def load_features(source, annotation=None, width=None, height=None, points=False):
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
					annotation_line = [int(i) for i in line[:-1].split(' ') if is_digit(i)]
					if len(annotation_line) > 4:
						annotation_line = annotation_line[-4:]

					annotations += [annotation_line]

			return np.array(annotations)
	
		def map_value(x, in_min, in_max, out_min, out_max):
			return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

		loader = AsyncVideoLoader(source=source, width=width, height=height).start()
		predictor = LandmarkPredictor()
		extractor = CheeksAndNose()

		features = np.empty([loader.lenght(), 3], dtype=np.float32)
		if annotation is not None:
			annotations = load_annotations(annotation)

		for i in range(loader.lenght()):
			success, frame = loader.read()
			# cv2.waitKey(1) & 0xFF
			def is_negative_annotation(annotation):
				for i in annotation:
					if i < 0:
						return True
				return False
			
			def is_null_annotation(annotation):
				for i in annotation:
					if i == 0:
						return True
				return False

			if (annotation is not None) and ((i >= len(annotations))
												or (is_null_annotation(annotations[i]))
												or (is_negative_annotation(annotations[i]))
												or (len(annotations[i]) != 4)):
				
				features[i] = 0 if i == 0 else features[i - 1]
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
					if width is not None:
						x, y, w, h = [map_value(x, 1920, 1080, width, height)
										for x in annotations[i]]
						
					if not points:
						frame = frame[y:y+h, x:x+w]
					else:
						frame = frame[y:h, x:w]

				landmarks = predictor.detect_landmarks(frame)
				frame = extractor.extract_roi(frame, landmarks).astype(np.float32)

				frame[frame == 0.0] = np.nan
				features[i] = np.nanmean(frame, axis=(0, 1))

		loader.stop()
		
		print("[DataLoader] Features extracted from '{0}'.".format(source))
		return features

class GenericDatasetLoader:
	@staticmethod
	def walk_and_load_from(root_folder, annotations_folder,	
										width=None, height=None, points=False):
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
							annotation=annotation,
							width=width,
							height=height,
							points=points)

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
		destination_files = os.listdir(destination)
		if "rad_train_real.npy" not in destination_files:	
			train_attack, train_real = ReplayAttackLoader.load_train(source)
			np.save("{0}/rad_train_fake.npy".format(destination), train_attack)
			np.save("{0}/rad_train_real.npy".format(destination), train_real)			
		
		if "rad_devel_real.npy" not in destination_files:
			devel_attack, devel_real = ReplayAttackLoader.load_devel(source)
			np.save("{0}/rad_devel_fake.npy".format(destination), devel_attack)
			np.save("{0}/rad_devel_real.npy".format(destination), devel_real)

		if "rad_test_real.npy" not in destination_files:
			test_attack, test_real = ReplayAttackLoader.load_test(source)
			np.save("{0}/rad_test_fake.npy".format(destination), test_attack)
			np.save("{0}/rad_test_real.npy".format(destination), test_real)


class SpoofInTheWildLoader:
	@staticmethod
	def __load_by_split(source, split):
		split_live = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/live".format(source, split),
			"{0}/{1}/live".format(source, split),
			width=1280,
			height=720,
			points=True
		)

		split_spoof = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/spoof".format(source, split),
			"{0}/{1}/spoof".format(source, split),
			width=1280,
			height=720,
			points=True
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

		np.save("{0}/siw_train_live.npy".format(destination), train_live)
		np.save("{0}/siw_train_spoof.npy".format(destination), train_spoof)

		np.save("{0}/siw_test_live.npy".format(destination), test_live)
		np.save("{0}/siw_test_spoof.npy".format(destination), test_spoof)


class OuluLoader:
	@staticmethod
	def __load_by_split(source, split):
		split_full = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}".format(source, split),
			None,
		)
		return split_full

	@staticmethod
	def load_train(source):
		return OuluLoader.__load_by_split(source, 'Train_files')

	@staticmethod
	def load_devel(source):
		return OuluLoader.__load_by_split(source, 'Dev_files')

	@staticmethod
	def load_test(source):
		return OuluLoader.__load_by_split(source, 'Test_files')

	@staticmethod
	def load_and_store(source, destination):
		train = OuluLoader.load_train(source)
		devel =  OuluLoader.load_devel(source)
		test = OuluLoader.load_test(source)

		np.save("{0}/oulu_train_full.npy".format(destination), train)
		np.save("{0}/oulu_devel_full.npy".format(destination), devel)
		np.save("{0}/oulu_test_full.npy".format(destination), test)

		#Store file names!

def get_args():
	parser = argparse.ArgumentParser(description="Código para extração das \
										bases de dados de PAD como SiW e \
										ReplayAttack")

	parser.add_argument("source", nargs=None, default=None, action="store")
	parser.add_argument("dest", nargs=None, default=None, action="store")
	parser.add_argument("time", nargs=None, default=5, action="store", type=int)

	return parser.parse_args()


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
			data_slice = np.append(data_slice, sl, axis = 0)

	return data_slice

if __name__ == "__main__":
	args = get_args()
	if args.source.endswith('ReplayAttack'):
		def slice_partition(name, size, stride, prefix='rad'):
			split_fake = np.load("{0}/rad_{1}_fake.npy".format(args.dest, name))
			split_real = np.load("{0}/rad_{1}_real.npy".format(args.dest, name))
			
			slice_split_fake = slice_and_stride(split_fake, size, stride)
			slice_split_real = slice_and_stride(split_real, size, stride)
			split = np.append(slice_split_fake, slice_split_real, axis=0)
			
			labels_split_fake = np.zeros([len(slice_split_fake)])
			labels_split_real = np.ones([len(slice_split_real)])
			labels = np.append(labels_split_fake, labels_split_real)
			
			np.save("{0}/{1}_{2}_x.npy".format(args.dest, prefix, name), split)
			np.save("{0}/{1}_{2}_y.npy".format(args.dest, prefix, name), labels)


		frame_rate = 24
		video_size = frame_rate * args.time
		if 'rad_train_real.npy' not in os.listdir(args.dest):
			ReplayAttackLoader.load_and_store(args.source, args.dest)

		slice_partition('train', video_size, 1)
		slice_partition('devel', video_size, 1)
		slice_partition('test', video_size, 1)

	elif args.source.endswith('SiW_release'):
		if 'siw_train_live.npy' not in os.listdir(args.dest):
			SpoofInTheWildLoader.load_and_store(args.source, args.dest)

		frame_rate = 30
		video_size = frame_rate * args.time
		
		protocol = 2
		if protocol == 1:
			def load_first_protocol(name):
				live = np.load('{0}/siw_{1}_live.npy'.format(args.dest, name))
				spoof = np.load('{0}/siw_{1}_spoof.npy'.format(args.dest, name))
				
				live = live[:, :60]
				spoof = spoof[:, :60]

				part = np.append(spoof, live, axis=0)
				labels = np.append(np.zeros(len(spoof)), np.ones(len(live)))

				np.save('{0}/siw_{1}_x.npy'.format(args.dest, name), part)
				np.save('{0}/siw_{1}_y.npy'.format(args.dest, name), labels)

			load_first_protocol('train')
			load_first_protocol('test')
		elif protocol == 2:
			#Aplicar data augmentation
			#(E não apenas truncar pelo tamanho máximo do vídeo)

			leave_out_medium = 4
			def load_hints(part):
				def parse_name(name):
					result = name.split('-')
					result[-1] = result[-1].split('.')[0]
					return [int(i) for i in result]

				names = list()
				with open(part, 'r') as file:
					for line in file:
						names.append(parse_name(line))

				return names

			train_x, test_x = None, None
			train_y, test_y = list(), list()
			def load_second_protocol(part, label):
				global train_x, test_x
				global train_y, test_y

				items = np.load('{0}/{1}.npy'.format(args.dest, part))
				hints = load_hints('{0}/{1}.txt'.format(args.dest, part))
				for i in range(len(items)):
					item = items[i][:video_size]
					if hints[i][3] == leave_out_medium:
						if test_x is None:
							test_x = item.reshape(1, video_size, 3)
						else:
							test_x = np.append(test_x, np.array([item]), axis=0)

						test_y.append(label)
					else:
						if train_x is None:
							train_x = item.reshape(1, video_size, 3)
						else:
							train_x = np.append(train_x, np.array([item]), axis=0)
						
						train_y.append(label)

			load_second_protocol('siw_train_spoof', 0)
			load_second_protocol('siw_train_live', 1)
			load_second_protocol('siw_test_spoof', 0)
			load_second_protocol('siw_test_live', 1)	

			np.save('{0}/siw_train_x.npy'.format(args.dest), np.array(train_x))
			np.save('{0}/siw_test_x.npy'.format(args.dest), np.array(test_x))
			
			np.save('{0}/siw_train_y.npy'.format(args.dest), np.array(train_y))
			np.save('{0}/siw_test_y.npy'.format(args.dest), np.array(test_y))
	elif args.source.endswith('Oulu_NPU'):
		if 'oulu_train_full.npy' not in os.listdir(args.dest):
			OuluLoader.load_and_store(args.source, args.dest)

		print("TODO")
	else:
		print("Base de dados não suportada.")
