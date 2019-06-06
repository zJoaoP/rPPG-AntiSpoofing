from utils.landmarks import LandmarkPredictor
from utils.extractor import CheeksAndNose
# from utils.loaders import 

import numpy as np
import cv2
import os

from keras.layers import Conv1D, Flatten, Dense, Input, Concatenate, GlobalAveragePooling1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.utils import to_categorical

# Carregar novamente os dados, usando DeHaan e DeHaanPOS (Ao invés de Green)
# DeHaan corrigido. Verificar resultados em vídeos falsos.

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
		network = Conv1D(base_filter_size, kernel_size = 11, strides = 1, activation = "relu", name = "conv_rgb1")(rgb_input)
		network = Conv1D(base_filter_size, kernel_size = 11, strides = 1, activation = "relu", name = "conv_rgb2")(network)
		network = MaxPooling1D(pool_size = 5, strides = 3, name = "max_pooling_rgb1")(network)

		network = Conv1D(2 * base_filter_size, kernel_size = 5, strides = 1, activation = "relu", name = "conv_rgb3")(network)
		network = Conv1D(2 * base_filter_size, kernel_size = 5, strides = 1, activation = "relu", name = "conv_rgb4")(network)

		network = GlobalAveragePooling1D(name = "gap_rgb")(network)

		return network

	def build_ppg_model(base_filter_size):
		network = Conv1D(base_filter_size, kernel_size = 11, strides = 1, activation = "relu", name = "conv_ppg1")(ppg_input)
		network = Conv1D(base_filter_size, kernel_size = 11, strides = 1, activation = "relu", name = "conv_ppg2")(network)
		network = MaxPooling1D(pool_size = 5, strides = 3, name = "max_pooling_ppg1")(network)

		network = Conv1D(2 * base_filter_size, kernel_size = 5, strides = 1, activation = "relu", name = "conv_ppg3")(network)
		network = Conv1D(2 * base_filter_size, kernel_size = 5, strides = 1, activation = "relu", name = "conv_ppg4")(network)

		network = GlobalAveragePooling1D(name = "gap_ppg")(network)

		return network

	def build_fft_model(base_filter_size):
		network = Conv1D(base_filter_size, kernel_size = 5, strides = 2, activation = "relu", name = "conv_fft1")(fft_input)
		network = Conv1D(base_filter_size, kernel_size = 5, strides = 2, activation = "relu", name = "conv_fft2")(network)
		network = MaxPooling1D(pool_size = 3, strides = 2, name = "max_pooling_fft1")(network)

		network = Conv1D(2 * base_filter_size, kernel_size = 3, strides = 2, activation = "relu", name = "conv_fft3")(network)
		network = Conv1D(2 * base_filter_size, kernel_size = 3, strides = 2, activation = "relu", name = "conv_fft4")(network)
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
		optimizer = "adam",
		loss = "binary_crossentropy",
		metrics = ["accuracy", FPR, FNR]
	)
	# combined.summary()
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
	time_series = np.empty([data.shape[0], data.shape[1], 2], dtype = np.float32)
	ffts = np.empty([data.shape[0], data.shape[1] // 2, 2], dtype = np.float32)

	for i in range(len(data)):
		if i > 0 and i % 500 == 0:
			print("[PostProcessing] {0} / {1} temporal series processed.".format(i, len(data)))

		pos_algorithm = DeHaanPOS(temporal_means = data[i])
		dehaan_algorithm = DeHaan(temporal_means = data[i])

		dehaan_ppg = dehaan_algorithm.measure_reference(frame_rate = frame_rate, window_size = int(frame_rate * 1.5)).reshape(data.shape[1], 1)
		pos_ppg = pos_algorithm.measure_reference(frame_rate = frame_rate, window_size = int(frame_rate * 1.5)).reshape(data.shape[1], 1)

		_, dehaan_fft = dehaan_algorithm.get_fft(dehaan_ppg, frame_rate = frame_rate)
		_, pos_fft = pos_algorithm.get_fft(pos_ppg, frame_rate = frame_rate)

		dehaan_fft = dehaan_fft.reshape(data.shape[1] // 2, 1)
		pos_fft = pos_fft.reshape(data.shape[1] // 2, 1)

		time_series[i] = np.append(dehaan_ppg, pos_ppg, axis = 1)
		ffts[i] = np.append(dehaan_fft, pos_fft, axis = 1)

	data = data / 255.0
	return [data, time_series, ffts]

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

		# ppg_train = np.load("{0}/train_ppg_x.npy".format(args.data_folder))
		# fft_train = np.load("{0}/train_fft_x.npy".format(args.data_folder))

		# ppg_validation = np.load("{0}/validation_ppg_x.npy".format(args.data_folder))
		# fft_validation = np.load("{0}/validation_fft_x.npy".format(args.data_folder))

		# t_x = [t_x, ppg_train, fft_train]
		# v_x = [v_x, ppg_validation, fft_validation]

	t_x = add_algorithm_info(t_x, frame_rate = frame_rate)
	v_x = add_algorithm_info(v_x, frame_rate = frame_rate)

	names = ["ppg_x", "fft_x"]
	for i, name in enumerate(names):
		np.save("{0}/train_{1}.npy".format(args.data_folder, name), t_x[1 + i])
		np.save("{0}/validation_{1}.npy".format(args.data_folder, name), v_x[1 + i])

	# t_x[0] = t_x[0] / np.max(t_x[0])
	# v_x[0] = v_x[0] / np.max(v_x[0])

	# from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, EarlyStopping
	# from itertools import product

	# model_prefix = "concat_model"
	# lr = [1e-3, 1e-4, 1e-5]
	# bs = [4, 8, 16, 32]
	# fs = [8, 16, 32]
	# for learning_rate, batch_size, base_filter_size in product(lr, bs, fs):
	# 	model_code = "lr_{0}_bs_{1}_bfs_{2}".format(learning_rate, batch_size, base_filter_size)
	# 	print("TRAINING {}! :)".format(model_code))
	
	# 	model_checkpoint = ModelCheckpoint("./models/{0}_{1}".format(model_prefix, model_code) + "_ep={epoch:02d}-loss={val_loss:.5f}-acc={val_acc:.4f}.hdf5", monitor = "val_acc", save_best_only = True, verbose = 1)
	# 	early_stopping = EarlyStopping(monitor = "val_acc", patience = 30, verbose = 1)
	# 	tensorboard = TensorBoard(log_dir = './logs/{0}/{1}/'.format(model_prefix, model_code))
		
	# 	model = build_model(frame_count = len(t_x[0][0]), learning_rate = learning_rate, base_filter_size = base_filter_size)

	# 	results = model.fit(
	# 		x = t_x,
	# 		y = t_y,
	# 		batch_size = batch_size,
	# 		epochs = 30000,
	# 		validation_data = (v_x, v_y),
	# 		callbacks = [tensorboard, model_checkpoint, early_stopping]
	# 	)