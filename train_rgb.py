'''
	Objetivos:
		1 - Escrever um gerador de batches com o objetivo de balancear as classes.
			- Fila circular
			- Random Sample
'''

from utils.landmarks import LandmarkPredictor
from utils.extractor import CheeksAndNose

import numpy as np
import cv2
import os

class VideoLoader:
	def __init__(self, time = 10, roi_extractor = CheeksAndNose()):
		self.roi_extractor = roi_extractor
		self.time = time

	def is_valid_video(self, filename):
		valid_extensions = ["avi", "mp4", "mov"]
		extension = filename.split('.')[-1]
		return extension in valid_extensions

	def get_video_info(self, filename):
		if not self.is_valid_video(filename):
			return -1, -1
		else:
			stream = cv2.VideoCapture(filename)
			frame_count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
			frame_rate = int(stream.get(cv2.CAP_PROP_FPS))
			stream.release()

			return frame_rate, frame_count

	def load_video(self, filename):
		if self.is_valid_video(filename):
			stream = cv2.VideoCapture(filename)
			face_predictor = LandmarkPredictor()
			frame_rate, current_frame = int(stream.get(cv2.CAP_PROP_FPS)), 0
			means = []

			while True:
				success, frame = stream.read()
				if not success:
					break

				face_rect = face_predictor.detect_face(image = frame)
				if face_rect is not None:
					face_left, face_right, face_top, face_bottom = face_rect.left(), face_rect.right(), face_rect.top(), face_rect.bottom()
					cut_face = frame[face_top : face_bottom, face_left : face_right, :]

					landmarks = face_predictor.detect_landmarks(image = cut_face)
					roi = self.roi_extractor.extract_roi(cut_face, landmarks)
					
					if current_frame % 60 == 0 and current_frame > 0:
						print("[VideoLoader] {0} frames extracted from '{1}'.".format(current_frame, filename))
					
					roi = roi.astype(np.float32)
					roi[roi < 5.0] = np.nan

					r = np.nanmean(roi[:, :, 2])
					g = np.nanmean(roi[:, :, 1])
					b = np.nanmean(roi[:, :, 0])

					means += [[r, g, b]]

					current_frame += 1
				else:
					print("[VideoLoader] Skipping '{0}'. Reason: Failure on Acquisition! (It's a folder??)".format(filename))
					return None

			return means
		else:
			print("[VideoLoader] Skipping '{0}'. Reason: Invalid Video! (It's a folder??)".format(filename))
			return None

class DataLoader:
	def __init__(self, folder, file_list = None, time = 15, num_channels = 3, roi_extractor = CheeksAndNose()):
		self.roi_extractor = roi_extractor
		self.num_channels = num_channels
		self.time = time

		self.file_list = file_list
		self.folder = folder

		self.loader = VideoLoader(time = time, roi_extractor = roi_extractor)

	def split_raw_data(self, raw_data, split_size, stride = 1):
		data, i = None, 0
		while i + split_size - 1 < len(raw_data):
			data = np.array([raw_data[i : i + split_size, :]]) if data is None else np.append(data, np.array([raw_data[i : i + split_size, :]]), axis = 0)
			i += stride

		return data

	def post_process(self, data, frame_rate = 25):
		min_data_size = min([len(data[i]) for i in range(len(data))])
		min_data_size = min(min_data_size, self.time * frame_rate)
		final_data = None

		for i in range(len(data)):
			sub_videos = self.split_raw_data(np.array(data[i]), split_size = min_data_size, stride = 1)
			final_data = sub_videos if final_data is None else np.append(final_data, sub_videos, axis = 0)

		return final_data

	def load_data(self):
		counter, data = 0, []
		dir_size = len(os.listdir(self.folder))
		for filename in sorted(os.listdir(self.folder)):
			counter += 1
			if (self.file_list is None) or ((self.file_list is not None) and (filename in self.file_list)):
				print("[DataLoader - {0} / {1}] Trying to load video '{2}'.".format(counter, dir_size, filename))
				video_data = self.loader.load_video(filename = self.folder + '/' + filename)
				if video_data is not None:
					data += [video_data]

		return self.post_process(data)

class ReplayAttackLoader:
	def custom_shuffle(self, a, b):
		assert len(a) == len(b)
		p = np.random.permutation(len(a))
		return a[p], b[p]

	def load_data(self, folder, time):
		def replace_array_by_cuts(raw_data_bucket, data_loader, split_size):
			new_data = None
			for i in range(len(raw_data_bucket)):
				split = data_loader.split_raw_data(raw_data_bucket[i], split_size, stride = 1)
				new_data = split if new_data is None else np.append(new_data, split, axis = 0)
			
			return new_data

		print("[ReplayAttackLoader] Loading ReplayAttack videos from '{0}/'.".format(folder))
		fake_fixed_loader = DataLoader(folder = folder + "/attack/fixed", time = time)
		fake_hand_loader = DataLoader(folder = folder + "/attack/hand", time = time)
		fake_fixed_data = fake_fixed_loader.load_data()
		fake_hand_data = fake_hand_loader.load_data()

		if fake_fixed_data.shape[1:] != fake_hand_data.shape[1:]:
			if fake_fixed_data.shape[1] < fake_hand_data.shape[1]:
				fake_hand_data = replace_array_by_cuts(fake_hand_data, fake_hand_loader, fake_fixed_data.shape[1])
			else:
				fake_fixed_data = replace_array_by_cuts(fake_fixed_data, fake_fixed_loader, fake_hand_data.shape[1])

		fake = np.append(fake_hand_data, fake_fixed_data, axis = 0)

		real_loader = DataLoader(folder = folder + "/real", time = time)
		real = real_loader.load_data()
		if fake.shape[1:] != real.shape[1:]:
			if fake.shape[1] < real.shape[1]:
				real = replace_array_by_cuts(real, real_loader, fake.shape[1])
			else:
				fake = replace_array_by_cuts(fake, real_loader, real.shape[1])

		labels = np.append(np.zeros(fake.shape[0]), np.ones(real.shape[0]))
		return self.custom_shuffle(np.append(fake, real, axis = 0), to_categorical(labels, 2))


from keras.layers import Conv1D, BatchNormalization, Activation, Input, Add, GlobalMaxPooling1D, Dense
from keras.utils import to_categorical
from keras.models import Model

from keras.optimizers import Adam

#Construir as métricas!
def build_cnn_model(input_shape):
	def resnet_block(input_layer, filters, name = "resnet_block_0", apply_activation = False):
		branch = input_layer

		branch = Conv1D(filters, kernel_size = 3, strides = 1, padding = 'same', use_bias = False, name = "{0}_Conv1D_1".format(name))(branch)
		branch = BatchNormalization(name = "{0}_BN_1".format(name))(branch)
		branch = Activation('relu', name = "{0}_ReLU_1".format(name))(branch)

		branch = Conv1D(filters, kernel_size = 3, strides = 1, padding = 'same', use_bias = False, name = "{0}_Conv1D_2".format(name))(branch)

		if apply_activation or (branch.shape[-1] != input_layer.shape[-1]):
			input_layer = Conv1D(filters, kernel_size = 3, strides = 1, padding = 'same', name = "{0}_Conv1D_Residual".format(name))(input_layer)

		branch = Add()([branch, input_layer])
		branch = BatchNormalization(name = "{0}_BN_F".format(name))(branch)
		branch = Activation('relu', name = "{0}_ReLU_F".format(name))(branch)

		return branch

	import keras.backend as K
	def FPR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_true, y_pred], axis = 0))

	def FNR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_pred, y_true], axis = 0))
	
	filters = [16, 32, 64, 128]
	
	resnet_input = Input(shape = input_shape)
	resnet = resnet_block(resnet_input, filters = 8, name = "resnet_block_0", apply_activation = True)
	for i in range(len(filters)):
		resnet = resnet_block(resnet, filters = filters[i], name = "resnet_block_{0}".format(i + 1), apply_activation = True)

	resnet = GlobalMaxPooling1D()(resnet)
	resnet = Dense(2, activation = 'softmax')(resnet)

	model = Model(inputs = resnet_input, outputs = resnet)
	model.compile(
		optimizer = Adam(lr = 1e-6),
		loss = "binary_crossentropy",
		metrics = ["accuracy", FNR, FPR]
	)

	model.summary()
	return model

import argparse

def build_parser():
	parser = argparse.ArgumentParser(description = "Código para treino das redes neurais utilizadas para detecção de ataques de apresentação.")
	parser.add_argument("dataset_folder", nargs = None, default = None, action = "store",
						help = "Localização da pasta do dataset. (Padrão: %(default)s)")
	parser.add_argument("process_folder", nargs = None, default = None, action = "store",
						help = "Localização da pasta onde serão (ou foram) armazenados os dados processados. (Padrão: %(default)s)")
	parser.add_argument("--time", dest = "time", nargs = "?", default = 5, action = "store", type = int,
						help = "Tempo (em segundos) de captura ou análise. (Padrão: %(default)s segundos)")
	parser.add_argument("--process", dest = "process_data", action = "store_true", default = False,
						help = "Define se serão obtidas novas médias a partir do dataset informado. (Padrão: %(default)s)")
	# parser.add_argument("--aproach", dest = "aproach", nargs = "?", default = ["DeHaan", "DeHaanPOS", "ICA"], action = "store",
	# 					help = "Abordagem no cálculo do rPPG. (Padrão: %(default)s)")
	return parser

from algorithms.de_haan_pos import DeHaanPOS

def add_algorithm_info(data, frame_rate):
	limit = 2000
	# data_after_processing = np.empty([data.shape[0], data.shape[1], data.shape[2] + 1])
	data_after_processing = np.empty([limit, data.shape[1], data.shape[2] + 1])
	for i in range(limit):
		if i > 0 and i % 500 == 0:
			print("[PostProcessing] {0} / {1} temporal series processed.".format(i, len(data)))

		rppg = DeHaanPOS(temporal_means = data[i]).measure_reference(frame_rate = frame_rate, window_size = int(frame_rate * 1.5))
		data_after_processing[i] = np.append(data[i], np.reshape(rppg, (rppg.shape[0], 1)), axis = 1)
	
	return data_after_processing
	
from keras.utils import Sequence	
class DataGenerator(Sequence):
	def __init__(self, train_x, train_y, batch_size = 32):
		self.batch_size = batch_size

		self.fake_x = np.array([train_x[i] for i in range(len(train_x)) if np.argmax(train_y[i]) == 0])
		self.real_x = np.array([train_x[i] for i in range(len(train_x)) if np.argmax(train_y[i]) == 1])
		self.real_indexes = np.arange(len(self.real_x))
		self.fake_indexes = np.arange(len(self.fake_x))

		for i in range(len(self.real_x)):
			self.real_x[i] = self.real_x[i] / np.mean(self.real_x[i])

		for i in range(len(self.fake_x)):
			self.fake_x[i] = self.fake_x[i] / np.mean(self.fake_x[i])

	def on_epoch_end(self):
		self.real_indexes = np.arange(len(self.real_x))
		self.fake_indexes = np.arange(len(self.fake_x))

	# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
	def shuffle(self, a, b):
		assert len(a) == len(b)
		p = np.random.permutation(len(a))
		return a[p], b[p]

	def __len__(self):
		return min(len(self.fake_x), len(self.real_x)) // self.batch_size

	def __getitem__(self, index):
		real_indexes = self.real_indexes[index : index + (self.batch_size // 2)]
		fake_indexes = self.fake_indexes[index : index + (self.batch_size // 2)]
		
		x = np.append(np.take(self.fake_x, fake_indexes, axis = 0), np.take(self.real_x, real_indexes, axis = 0), axis = 0)
		y = np.append(np.zeros(self.batch_size // 2), np.ones(self.batch_size // 2))

		return self.shuffle(x, to_categorical(y, 2))

if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

	t_x, t_y, v_x, v_y = None, None, None, None
	frame_rate = 25 #Obter frame-rate a partir do dataset. Solução temporária.
	
	args = build_parser().parse_args()
	if args.process_data is True:
		loader = ReplayAttackLoader()
		t_x, t_y = loader.load_data(folder = "{0}/train".format(args.dataset_folder), time = args.time)
		v_x, v_y = loader.load_data(folder = "{0}/devel".format(args.dataset_folder), time = args.time)

		np.save("{0}/train_x.npy".format(args.process_folder), t_x)
		np.save("{0}/train_y.npy".format(args.process_folder), t_y)

		np.save("{0}/validation_x.npy".format(args.process_folder), v_x)
		np.save("{0}/validation_y.npy".format(args.process_folder), v_y)
	else:
		t_x = np.load("{0}/train_x.npy".format(args.process_folder))
		t_y = np.load("{0}/train_y.npy".format(args.process_folder))
		v_x = np.load("{0}/validation_x.npy".format(args.process_folder))
		v_y = np.load("{0}/validation_y.npy".format(args.process_folder))

	# print("Total de exemplos positivos nos dados de treino: {0}".format(np.sum(np.argmax(t_y, axis = 1))))
	# print("Tamanho dos dados de treino: {0}".format(len(t_y)))

	print("Train shape: {0}".format(t_x.shape))
	print("Validation shape: {0}".format(v_x.shape))

	model = build_cnn_model(input_shape = t_x[0].shape)

	validation_generator = DataGenerator(v_x, v_y, batch_size = 8)
	train_generator = DataGenerator(t_x, t_y, batch_size = 8)
	model.fit_generator(epochs = 100, generator = train_generator, validation_data = validation_generator)