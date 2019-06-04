from utils.landmarks import LandmarkPredictor
from utils.extractor import CheeksAndNose
# from utils.loaders import 

import numpy as np
import cv2
import os

from keras.layers import Dense, BatchNormalization, Activation, MaxPooling1D, Dropout, Conv1D, Reshape, GlobalAveragePooling1D
from keras.utils import to_categorical
from keras.models import Model, Sequential

def build_conv_model(input_shape, learning_rate = 2e-3):
	import keras.backend as K

	def FPR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_true, y_pred], axis = 0))

	def FNR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_pred, y_true], axis = 0))

	from keras.regularizers import l1, l2

	model = Sequential()
	
	# model.add(Conv1D(128, kernel_size = 11, strides = 1, use_bias = False, input_shape = input_shape))
	# model.add(BatchNormalization())
	# model.add(Activation("relu"))

	# model.add(Conv1D(128, kernel_size = 11, strides = 1, use_bias = False))
	# model.add(BatchNormalization())
	# model.add(Activation("relu"))
	
	# model.add(MaxPooling1D(pool_size = 3, strides = 2))

	# model.add(Conv1D(256, kernel_size = 11, strides = 1, use_bias = False))
	# model.add(BatchNormalization())
	# model.add(Activation("relu"))

	# model.add(Conv1D(256, kernel_size = 11, strides = 1, use_bias = False))
	# model.add(BatchNormalization())
	# model.add(Activation("relu"))

	# model.add(GlobalAveragePooling1D())
	# model.add(Dropout(0.5))

	from keras.layers import LSTM, Flatten

	model.add(Reshape(target_shape = (23, 10), input_shape = input_shape))
	model.add(LSTM(128, return_sequences = True))
	model.add(Flatten())
	model.add(Dense(2, activation = "softmax"))

	from keras.optimizers import Adam
	model.compile(
		optimizer = Adam(lr = learning_rate),
		loss = "binary_crossentropy",
		metrics = ["accuracy", FPR, FNR]
	)

	model.summary()

	return model

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

from algorithms.green import Green

def add_algorithm_info(data, frame_rate):
	data_after_processing = np.empty([data.shape[0], data.shape[1], 5], dtype = np.float32)
	Upsampling1D = lambda x : np.array([list([e, e]) for e in x]).flatten() # (Upsampling1D)

	for i in range(len(data)):
		if i > 0 and i % 500 == 0:
			print("[PostProcessing] {0} / {1} temporal series processed.".format(i, len(data)))

		algorithm = Green(green_means = data[i][:, 1])
		rppg = algorithm.measure_reference(frame_rate = frame_rate, window_size = int(frame_rate * 1.6))
		_, y = algorithm.get_fft(rppg, frame_rate = frame_rate)

		y = Upsampling1D(y)

		data[i] = data[i] / 255.0
		data_with_ppg = np.append(data[i], rppg.reshape(rppg.shape[0], 1), axis = 1)
		data_after_processing[i] = np.append(data_with_ppg, y.reshape(data[0].shape[0], 1), axis = 1)
	
	return data_after_processing
	
from keras.utils import Sequence
class DataGenerator(Sequence):
	def __init__(self, train_x, train_y, batch_size = 32, noise_weight = 1e-2):
		self.batch_size = batch_size

		self.fake_x = np.array([train_x[i] for i in range(len(train_x)) if np.argmax(train_y[i]) == 0])
		self.real_x = np.array([train_x[i] for i in range(len(train_x)) if np.argmax(train_y[i]) == 1])

		self.real_indexes = np.arange(len(self.real_x))
		self.fake_indexes = np.arange(len(self.fake_x))
		self.real_pos, self.fake_pos = 0, 0
		self.noise_weight = noise_weight

	# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
	def shuffle(self, a, b):
		assert len(a) == len(b)
		p = np.random.permutation(len(a))
		return a[p], b[p]

	# def data_augmentation(self, batch):
	# 	for i in range(len(batch)):
	# 		flip_prob, scale_prob, offset_prob, noise_prob = np.random.random_sample(size = 4)
	# 		if flip_prob > 0.5:
	# 			batch[i] = np.flip(batch[i], axis = 0)
	# 		if noise_prob > 0.5:				
	# 			batch[i] = batch[i] + self.noise_weight * np.random.standard_normal(size = batch[i].shape)
	# 		if offset_prob > 0.5:
	# 			random_offset = np.random.random_sample() / 10.0
	# 			random_offset = random_offset if np.random.random_sample() < 0.5 else -random_offset
	# 			batch[i] = batch[i] + random_offset
	# 		# if scale_prob > 0.5:
	# 		# 	batch[i] = batch[i] * (1.0 + np.random.random_sample())

	# 	return batch

	def __len__(self):
		return min(len(self.fake_x), len(self.real_x)) // self.batch_size

	def __getitem__(self, index):
		if self.real_pos + (self.batch_size // 2) > len(self.real_x):
			self.real_indexes = np.arange(len(self.real_x))
			self.real_pos = 0

		if self.fake_pos + (self.batch_size // 2) > len(self.fake_x):
			self.fake_indexes = np.arange(len(self.fake_x))
			self.fake_pos = 0

		real_indexes = self.real_indexes[self.real_pos : self.real_pos + (self.batch_size // 2)]
		fake_indexes = self.fake_indexes[self.fake_pos : self.fake_pos + (self.batch_size // 2)]
		
		self.real_pos += self.batch_size // 2
		self.fake_pos += self.batch_size // 2
		
		x = np.append(np.take(self.fake_x, fake_indexes, axis = 0), np.take(self.real_x, real_indexes, axis = 0), axis = 0)
		y = np.append(np.zeros(self.batch_size // 2), np.ones(self.batch_size // 2))

		# x = self.data_augmentation(x)
		return self.shuffle(x, to_categorical(y, 2))

if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

	t_x, t_y, v_x, v_y = None, None, None, None
	frame_rate = 25 #Obter frame-rate a partir do dataset. Solução temporária.
	
	args = build_parser().parse_args()
	if args.process_data is True:
		loader = ReplayAttackLoader()
		t_x, t_y = loader.load_data(folder = "{0}/train".format(args.data_folder), time = args.time)
		v_x, v_y = loader.load_data(folder = "{0}/devel".format(args.data_folder), time = args.time)

		np.save("{0}/train_x.npy".format(args.data_folder), t_x)
		np.save("{0}/train_y.npy".format(args.data_folder), t_y)

		np.save("{0}/validation_x.npy".format(args.data_folder), v_x)
		np.save("{0}/validation_y.npy".format(args.data_folder), v_y)

		t_x = add_algorithm_info(t_x, frame_rate)
		v_x = add_algorithm_info(v_x, frame_rate)

		np.save("{0}/algo_train_x.npy".format(args.data_folder), t_x)
		np.save("{0}/algo_validation_x.npy".format(args.data_folder), v_x)
	else:
		t_x = np.load("{0}/train_x.npy".format(args.data_folder))
		t_y = np.load("{0}/train_y.npy".format(args.data_folder))

		v_x = np.load("{0}/validation_x.npy".format(args.data_folder))
		v_y = np.load("{0}/validation_y.npy".format(args.data_folder))

		t_x = np.load("{0}/fft_green_train_x.npy".format(args.data_folder))
		v_x = np.load("{0}/fft_green_validation_x.npy".format(args.data_folder))

	print("Total de exemplos positivos nos dados de treino: {0}".format(np.sum(np.argmax(t_y, axis = 1))))
	print("Tamanho dos dados de treino: {0}".format(len(t_y)))

	print("Train shape: {0}".format(t_x.shape))
	print("Validation shape: {0}".format(v_x.shape))

	# t_x = add_algorithm_info(t_x, frame_rate)
	# v_x = add_algorithm_info(v_x, frame_rate)

	# np.save("{0}/fft_green_train_x.npy".format(args.data_folder), t_x)
	# np.save("{0}/fft_green_validation_x.npy".format(args.data_folder), v_x)

	t_x = t_x[:, :, 3].reshape(t_x.shape[0], t_x.shape[1], 1)
	v_x = v_x[:, :, 3].reshape(v_x.shape[0], v_x.shape[1], 1)

	from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

	model_name = "lstm_ppg_lr=1e-4_bs=8"
	train_generator = DataGenerator(t_x, t_y, batch_size = 8)
	
	model_checkpoint = ModelCheckpoint("./models/{0}".format(model_name) + "_ep={epoch:02d}-loss={val_loss:.5f}-acc={val_acc:.4f}.hdf5", monitor = "val_loss", save_best_only = True, verbose = 1)
	tensorboard = TensorBoard(log_dir = './logs/lstm_ppg/{0}/'.format(model_name))
	# early_stopping = EarlyStopping(monitor = "val_loss", patience = 1500, verbose = 1, min_delta = 0.0001)
	
	model = build_conv_model(input_shape = t_x[0].shape, learning_rate = 1e-4)
	
	results = model.fit_generator(
		epochs = 30000,
		generator = train_generator,
		validation_data = (v_x, v_y),
		callbacks = [model_checkpoint, tensorboard]
	)

	# model = build_cnn_model(input_shape = t_x[0].shape, learning_rate = 1.0)
	# results = []
	# for file_name in os.listdir("./models/"):
	# 	model.load_weights("./models/{0}".format(file_name))
	# 	results += [[file_name] + model.evaluate(v_x, v_y)]

	# print(model.metrics_names)
	# for r in sorted(results, key = lambda x : x[1]):
	# 	print(r)