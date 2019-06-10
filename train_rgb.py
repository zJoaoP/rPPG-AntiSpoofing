from utils.landmarks import LandmarkPredictor
from utils.extractor import CheeksAndNose
# from utils.loaders import 

import numpy as np
import cv2
import os

from keras.layers import Flatten, Input, Concatenate, Dropout, MaxPooling1D, Activation, GlobalAveragePooling1D
from keras.layers import Conv1D, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical

from keras.optimizers import Adam

def build_model(frame_count, learning_rate = 1e-4, base_filter_size = 8):
	import keras.backend as K

	def FPR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_true, y_pred], axis = 0))

	def FNR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_pred, y_true], axis = 0))

	def build_rgb_model(base_filter_size):		
		network = Conv1D(base_filter_size, kernel_size = 11, strides = 1, use_bias = False, name = "conv_rgb1")(rgb_input)
		network = BatchNormalization(name = "batch_normalization_rgb1")(network)
		network = Activation("relu", name = "activation_rgb1")(network)
		network = MaxPooling1D(pool_size = 5, strides = 3, name = "max_pooling_rgb1")(network)

		for i in range(2):
			network = Conv1D((2 ** (1 + i)) * base_filter_size, kernel_size = 7, strides = 1, use_bias = False, name = "conv_rgb{}".format(i + 2))(network)
			network = BatchNormalization(name = "batch_normalization_rgb{}".format(i + 2))(network)
			network = Activation("relu", name = "activation_rgb{}".format(i + 2))(network)
			network = MaxPooling1D(pool_size = 3, strides = 2, name = "max_pooling_rgb{}".format(i + 2))(network)

		network = GlobalAveragePooling1D(name = "gap_rgb")(network)

		return network

	def build_ppg_model(base_filter_size):
		network = Conv1D(base_filter_size, kernel_size = 11, strides = 1, use_bias = False, name = "conv_ppg1")(ppg_input)
		network = BatchNormalization(name = "batch_normalization_ppg1")(network)
		network = Activation("relu", name = "activation_ppg1")(network)
		network = MaxPooling1D(pool_size = 5, strides = 3, name = "max_pooling_ppg1")(network)

		for i in range(2):
			network = Conv1D((2 ** (1 + i)) * base_filter_size, kernel_size = 7, strides = 1, use_bias = False, name = "conv_ppg{}".format(i + 2))(network)
			network = BatchNormalization(name = "batch_normalization_ppg{}".format(i + 2))(network)
			network = Activation("relu", name = "activation_ppg{}".format(i + 2))(network)
			network = MaxPooling1D(pool_size = 3, strides = 2, name = "max_pooling_ppg{}".format(i + 2))(network)

		network = GlobalAveragePooling1D(name = "gap_ppg")(network)

		return network

	def build_fft_model(base_filter_size):
		network = Conv1D(base_filter_size, kernel_size = 5, strides = 2, use_bias = False, name = "conv_fft1")(fft_input)
		network = BatchNormalization(name = "batch_normalization_fft1")(network)
		network = Activation("relu", name = "activation_fft1")(network)
		network = MaxPooling1D(pool_size = 3, strides = 2, name = "max_pooling_fft1")(network)

		for i in range(2):
			network = Conv1D((2 ** (1 + i)) * base_filter_size, kernel_size = 3, strides = 1, use_bias = False, name = "conv_fft{}".format(i + 2))(network)
			network = BatchNormalization(name = "batch_normalization_fft{}".format(i + 2))(network)
			network = Activation("relu", name = "activation_fft{}".format(i + 2))(network)
			network = MaxPooling1D(pool_size = 3, strides = 2, name = "max_pooling_fft{}".format(i + 2))(network)

		network = GlobalAveragePooling1D(name = "gap_fft")(network)

		return network

	rgb_input = Input(shape = (frame_count, 3), name = "rgb_input")
	ppg_input = Input(shape = (frame_count, 2), name = "ppg_input")
	fft_input = Input(shape = (frame_count // 2, 2), name = "fft_input")
	
	build_functions = [build_rgb_model, build_ppg_model, build_fft_model]
	rgb_model, ppg_model, fft_model = [call(base_filter_size) for call in build_functions]

	combined = Concatenate(name = "merge_combined")([rgb_model, ppg_model, fft_model])
	combined = Dropout(rate = 0.5, name = "dropout_combined1")(combined)

	combined = Dense(2, activation = "softmax", name = "final_layer")(combined)

	combined = Model(inputs = [rgb_input, ppg_input, fft_input], outputs = combined)

	combined.compile(
		optimizer = Adam(lr = learning_rate),
		loss = "binary_crossentropy",
		metrics = ["accuracy", FPR, FNR]
	)
	combined.summary()
	return combined

import argparse

def build_parser():
	parser = argparse.ArgumentParser(description = "Código para treino das redes neurais utilizadas para detecção de ataques de apresentação.")
	parser.add_argument("data_folder", nargs = None, default = None, action = "store",
						help = "Localização da pasta do dataset (ou dos arquivos pré-processados). (Padrão: %(default)s)")
	parser.add_argument("--time", dest = "time", nargs = "?", default = 5, action = "store", type = int,
						help = "Tempo (em segundos) de captura ou análise. (Padrão: %(default)s segundos)")
	parser.add_argument("--process", dest = "process_data", action = "store_true", default = False,
						help = "Define se serão obtidas novas médias a partir do dataset informado. (Padrão: %(default)s)")
	return parser

from algorithms.de_haan_pos import DeHaanPOS
from algorithms.de_haan import DeHaan

def add_algorithm_info(data, frame_rate):
	from copy import deepcopy
	time_series = np.empty([data.shape[0], data.shape[1], 2], dtype = np.float32)
	ffts = np.empty([data.shape[0], data.shape[1] // 2, 2], dtype = np.float32)

	for i in range(len(data)):
		if i > 0 and i % 128 == 0:
			print("[PostProcessing] {0} / {1} temporal series processed.".format(i, len(data)))

		pos_algorithm = DeHaanPOS(temporal_means = deepcopy(data[i]))
		dehaan_algorithm = DeHaan(temporal_means = deepcopy(data[i]))

		dehaan_ppg = dehaan_algorithm.measure_reference(frame_rate = frame_rate, window_size = int(frame_rate * 1.6))
		pos_ppg = pos_algorithm.measure_reference(frame_rate = frame_rate, window_size = int(frame_rate * 1.6))

		x_dehaan, dehaan_fft = dehaan_algorithm.get_fft(dehaan_ppg, frame_rate = frame_rate)		
		x_pos, pos_fft = pos_algorithm.get_fft(pos_ppg, frame_rate = frame_rate)

		dehaan_ppg = dehaan_ppg.reshape(data.shape[1], 1)
		pos_ppg = pos_ppg.reshape(data.shape[1], 1)

		dehaan_fft = dehaan_fft.reshape(data.shape[1] // 2, 1)
		pos_fft = pos_fft.reshape(data.shape[1] // 2, 1)

		time_series[i] = np.append(dehaan_ppg, pos_ppg, axis = 1)
		ffts[i] = np.append(dehaan_fft, pos_fft, axis = 1)

		data[i] = data[i] / np.max(data[i])

	return [data, time_series, ffts]

from keras.utils import Sequence
class DataGenerator(Sequence):
	def __init__(self, train_x, train_y, batch_size = 32, noise_weight = 0.01):
		self.batch_size = batch_size

		self.train_x = train_x
		self.train_y = train_y

		self.fake_positions_x = np.array([i for i in range(len(self.train_x[0])) if np.argmax(train_y[i]) == 0])
		self.real_positions_x = np.array([i for i in range(len(self.train_x[0])) if np.argmax(train_y[i]) == 1])

		self.real_indexes = np.arange(len(self.real_positions_x))
		self.fake_indexes = np.arange(len(self.fake_positions_x))

		self.real_pos, self.fake_pos = 0, 0
		self.noise_weight = noise_weight

	def __len__(self):
		return min(len(self.fake_positions_x), len(self.real_positions_x)) // self.batch_size

	def data_augmentation(self, x, threshold = 0.3, frame_rate = 25):
		augmented_data = x
		for i in range(len(x)):
			coin, coin_global_noise, coin_color_noise, coin_frame_rate = [np.random.sample() for x in range(4)]
			if coin > threshold:
				x_means, x_ppg, x_fft = [x[j][i] for j in range(3)]
				sample_frame_rate = frame_rate
				if coin_frame_rate > threshold:
					sample_frame_rate = int(sample_frame_rate + 5 * np.random.sample())
					sample_frame_rate = int(sample_frame_rate - 5 * np.random.sample())

				if coin_global_noise > threshold:
					x_means += self.noise_weight * np.random.normal(size = x_means.shape)
					x_ppg += self.noise_weight * np.random.normal(size = x_ppg.shape)
					x_fft += self.noise_weight * np.random.normal(size = x_fft.shape)				
				elif coin_color_noise > threshold:
					from copy import deepcopy

					x_means += self.noise_weight * np.random.normal(size = x_means.shape)
					dehaan_algorithm = DeHaan(temporal_means = deepcopy(x_means))
					pos_algorithm = DeHaanPOS(temporal_means = deepcopy(x_means))

					x_dehaan_ppg = dehaan_algorithm.measure_reference(frame_rate = sample_frame_rate, window_size = int(sample_frame_rate * 1.6))
					x_pos_ppg = pos_algorithm.measure_reference(frame_rate = sample_frame_rate, window_size = int(sample_frame_rate * 1.6))

					_, x_dehaan_fft = dehaan_algorithm.get_fft(x_dehaan_ppg, frame_rate = sample_frame_rate)		
					_, x_pos_fft = pos_algorithm.get_fft(x_pos_ppg, frame_rate = sample_frame_rate)

					x_dehaan_ppg = x_dehaan_ppg.reshape(x_means.shape[0], 1)
					x_pos_ppg = x_pos_ppg.reshape(x_means.shape[0], 1)

					x_dehaan_fft = x_dehaan_fft.reshape(x_means.shape[0] // 2, 1)
					x_pos_fft = x_pos_fft.reshape(x_means.shape[0] // 2, 1)

					x_ppg = np.append(x_dehaan_ppg, x_pos_ppg, axis = 1)
					x_fft = np.append(x_dehaan_fft, x_pos_fft, axis = 1)

				augmented_data[:][i] = [x_means, x_ppg, x_fft]
		return augmented_data


	def __getitem__(self, index):
		if self.real_pos + (self.batch_size // 2) > len(self.real_positions_x):
			self.real_indexes = np.arange(len(self.real_positions_x))
			self.real_pos = 0

		if self.fake_pos + (self.batch_size // 2) > len(self.fake_positions_x):
			self.fake_indexes = np.arange(len(self.fake_positions_x))
			self.fake_pos = 0

		real_indexes = self.real_indexes[self.real_pos : self.real_pos + (self.batch_size // 2)]
		fake_indexes = self.fake_indexes[self.fake_pos : self.fake_pos + (self.batch_size // 2)]

		real_indexes = np.take(self.real_positions_x, real_indexes, axis = 0)
		fake_indexes = np.take(self.fake_positions_x, fake_indexes, axis = 0)
		
		self.real_pos += self.batch_size // 2
		self.fake_pos += self.batch_size // 2

		x = [np.append(np.take(self.train_x[i], fake_indexes, axis = 0), np.take(self.train_x[i], real_indexes, axis = 0), axis = 0) for i in range(3)]
		y = np.append(np.zeros(self.batch_size // 2), np.ones(self.batch_size // 2))

		return self.data_augmentation(x), to_categorical(y, 2)

if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	frame_rate = 25
	
	args = build_parser().parse_args()
	if args.process_data is True:
		loader = ReplayAttackLoader()
		t_x, t_y = loader.load_data(folder = "{0}/train".format(args.data_folder), time = args.time)
		v_x, v_y = loader.load_data(folder = "{0}/devel".format(args.data_folder), time = args.time)

		np.save("{0}/train_x.npy".format(args.data_folder), t_x)
		np.save("{0}/train_y.npy".format(args.data_folder), t_y)

		np.save("{0}/validation_x.npy".format(args.data_folder), v_x)
		np.save("{0}/validation_y.npy".format(args.data_folder), v_y)

		t_x = add_algorithm_info(t_x, frame_rate = frame_rate)
		v_x = add_algorithm_info(v_x, frame_rate = frame_rate)

		names = ["ppg_x", "fft_x"]
		for i, name in enumerate(names):
			np.save("{0}/train_{1}.npy".format(args.data_folder, name), t_x[1 + i])
			np.save("{0}/validation_{1}.npy".format(args.data_folder, name), v_x[1 + i])
	else:
		t_x = np.load("{0}/train_x.npy".format(args.data_folder))
		t_y = np.load("{0}/train_y.npy".format(args.data_folder))

		v_y = np.load("{0}/validation_y.npy".format(args.data_folder))
		v_x = np.load("{0}/validation_x.npy".format(args.data_folder))

		ppg_train = np.load("{0}/train_ppg_x.npy".format(args.data_folder))
		fft_train = np.load("{0}/train_fft_x.npy".format(args.data_folder))

		ppg_validation = np.load("{0}/validation_ppg_x.npy".format(args.data_folder))
		fft_validation = np.load("{0}/validation_fft_x.npy".format(args.data_folder))

		t_x = [t_x, ppg_train, fft_train]
		v_x = [v_x, ppg_validation, fft_validation]

	from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, EarlyStopping
	from keras.utils import plot_model
	from itertools import product

	model_prefix = "concat_model"
	lr = [1e-4, 1e-5, 3e-5]
	bs = [8, 16, 32]
	fs = [16, 32]
	for batch_size, base_filter_size, learning_rate in product(bs, fs, lr):
		model_code = "lr_{0}_bs_{1}_bfs_{2}".format(learning_rate, batch_size, base_filter_size)
		print("TRAINING {}! :)".format(model_code))
	
		model_checkpoint = ModelCheckpoint("./models/{0}_{1}".format(model_prefix, model_code) + "_ep={epoch:02d}-loss={val_loss:.5f}-acc={val_acc:.4f}.hdf5", monitor = "val_acc", save_best_only = True, verbose = 1)
		early_stopping = EarlyStopping(monitor = "val_acc", patience = 1000, verbose = 1)
		tensorboard = TensorBoard(log_dir = './logs/{0}/{1}/'.format(model_prefix, model_code))

		model = build_model(frame_count = len(t_x[0][0]), learning_rate = learning_rate, base_filter_size = base_filter_size)

		train_generator = DataGenerator(t_x, t_y, batch_size = batch_size)

		results = model.fit_generator(
			generator = train_generator,
			epochs = 30000,
			validation_data = (v_x, v_y),
			callbacks = [tensorboard, model_checkpoint, early_stopping]
		)