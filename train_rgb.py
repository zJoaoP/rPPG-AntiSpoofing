from utils.landmarks import LandmarkPredictor
from utils.extractor import CheeksAndNose
# from utils.loaders import 

import numpy as np
import cv2
import os

from keras.layers import Dense, BatchNormalization, Activation, Reshape, Dropout
from keras.utils import to_categorical
from keras.models import Model, Sequential

from keras.optimizers import Adam

def build_cnn_model(input_shape, learning_rate = 2e-3):
	import keras.backend as K

	def FPR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_true, y_pred], axis = 0))

	def FNR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_pred, y_true], axis = 0))

	def residual_block(input_layer, filters, kernel_size, name = "residual_without_name"):
		y = input_layer

		y = Conv1D(filters, kernel_size = kernel_size, padding = 'same', name = "{0}_Conv1D_1".format(name))(input_layer)
		y = Activation("relu")(y)
		y = Conv1D(filters, kernel_size = kernel_size, padding = 'same', name = "{0}_Conv1D_2".format(name))(y)

		y = Add()([y, input_layer])
		y = Activation("relu")(y)
		return y

	def pooling_residual_block(input_layer, filters, kernel_size, name = "residual_without_name"):
		y = input_layer

		y = Conv1D(filters, kernel_size = kernel_size, strides = 2, padding = 'same', name = "{0}_Conv1D_1".format(name))(input_layer)
		y = Activation("relu")(y)

		y = Conv1D(filters, kernel_size = kernel_size, padding = 'same', name = "{0}_Conv1D_2".format(name))(y)

		input_layer = Conv1D(filters, kernel_size = kernel_size, strides = 2, padding = 'same', name = "{0}_Residual".format(name))(input_layer)

		y = Add()([y, input_layer])

		y = Activation("relu")(y)
		return y

	model = Sequential()
	model.add(Reshape(target_shape = (230 * 3,), input_shape = (230, 3)))
	
	model.add(Dense(256, use_bias = False))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Activation("relu"))

	model.add(Dense(2, use_bias = False))
	model.add(BatchNormalization())
	model.add(Activation("softmax"))

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
	data_after_processing = np.empty([data.shape[0], data.shape[1], 4], dtype = np.float32)
	for i in range(len(data)):
		if i > 0 and i % 500 == 0:
			print("[PostProcessing] {0} / {1} temporal series processed.".format(i, len(data)))

		algorithm = Green(green_means = data[i][:, 1])
		rppg = algorithm.measure_reference(frame_rate = frame_rate, window_size = int(frame_rate * 1.6))
		rppg = (rppg - np.min(rppg)) / (np.max(rppg) - np.min(rppg))

		data[i] = data[i] / 255.0
		data_after_processing[i] = np.append(data[i], rppg.reshape(rppg.shape[0], 1), axis = 1)
	
	return data_after_processing
	
from keras.utils import Sequence
class DataGenerator(Sequence):
	def __init__(self, train_x, train_y, batch_size = 32, noise_weight = 0.001):
		self.batch_size = batch_size

		self.fake_x = np.array([train_x[i] for i in range(len(train_x)) if np.argmax(train_y[i]) == 0])
		self.real_x = np.array([train_x[i] for i in range(len(train_x)) if np.argmax(train_y[i]) == 1])

		self.fake_x = np.append(self.fake_x, np.flip(self.fake_x, axis = 0), axis = 0)
		self.real_x = np.append(self.real_x, np.flip(self.real_x, axis = 0), axis = 0)

		self.real_indexes = np.arange(len(self.real_x))
		self.fake_indexes = np.arange(len(self.fake_x))
		self.real_pos, self.fake_pos = 0, 0
		self.noise_weight = noise_weight

	# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
	def shuffle(self, a, b):
		assert len(a) == len(b)
		p = np.random.permutation(len(a))
		return a[p], b[p]

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

		# t_x = np.load("{0}/algo_train_x.npy".format(args.data_folder))
		# v_x = np.load("{0}/algo_validation_x.npy".format(args.data_folder))

	print("Total de exemplos positivos nos dados de treino: {0}".format(np.sum(np.argmax(t_y, axis = 1))))
	print("Tamanho dos dados de treino: {0}".format(len(t_y)))

	print("Train shape: {0}".format(t_x.shape))
	print("Validation shape: {0}".format(v_x.shape))

	from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
	from itertools import product
	
	lr = [1e-4, 3e-4, 1e-5, 3e-5]
	bs = [4, 8, 16, 32, 64]

	t_x = t_x / 255.0
	v_x = v_x / 255.0

	for batch_size, learning_rate in product(bs, lr):
		model_name = "dense_rgb_lr={0}_bs={1}".format(learning_rate, batch_size)
		train_generator = DataGenerator(t_x, t_y, batch_size = batch_size)
		
		model_checkpoint = ModelCheckpoint("./models/{0}".format(model_name) + "_ep={epoch:02d}-loss={val_loss:.5f}-acc={val_acc:.4f}.hdf5", monitor = "val_loss", save_best_only = True, verbose = 1)
		tensorboard = TensorBoard(log_dir = './logs/dense_rgb_ppg/{0}/'.format(model_name))
		early_stopping = EarlyStopping(monitor = "val_loss", patience = 300, verbose = 1)
		
		model = build_cnn_model(input_shape = t_x[0].shape, learning_rate = learning_rate)
		
		results = model.fit_generator(
			epochs = 10000,
			generator = train_generator,
			validation_data = (v_x, v_y),
			callbacks = [model_checkpoint, tensorboard, early_stopping]
		)