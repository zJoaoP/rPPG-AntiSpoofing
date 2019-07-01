import keras.backend as K
import numpy as np
import cv2
import os

from keras.layers import Flatten, Input, Concatenate, Dropout, MaxPooling1D, Activation, GlobalAveragePooling1D
from keras.layers import Conv1D, Dense, Dropout, BatchNormalization, Lambda, Add
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam


def build_concat_model(frame_count, learning_rate = 1e-4, base_filter_size = 8):
	def FPR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_true, y_pred], axis = 0))

	def FNR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_pred, y_true], axis = 0))

	def build_resnet_identity_block(input_layer, filters, kernel_size, name = "resnet_identity_block"):
		block_layer = Conv1D(filters, kernel_size = kernel_size, strides = 1, padding = "same", name = "{0}_Conv1D_1".format(name))(input_layer)
		block_layer = Activation("relu", name = "{0}_ReLU_1".format(name))(block_layer)
		block_layer = Conv1D(filters, kernel_size = kernel_size, strides = 1, padding = "same", name = "{0}_Conv1D_2".format(name))(block_layer)
		
		block_layer = Add(name = "{0}_Identity_Add".format(name))([block_layer, input_layer])
		
		return Activation("relu", name = "{0}_ReLU_2".format(name))(block_layer)

	def build_resnet_pooling_block(input_layer, filters, kernel_size, name = "resnet_pooling_block"):
		block_layer = Conv1D(filters, kernel_size = kernel_size, padding = "same", strides = 1, name = "{0}_Conv1D_1".format(name))(input_layer)
		block_layer = Activation("relu", name = "{0}_ReLU_1".format(name))(block_layer)
		block_layer = Conv1D(filters, kernel_size = kernel_size, padding = "same", strides = 1, name = "{0}_Conv1D_2".format(name))(block_layer)
		
		conv_layer = Conv1D(filters, kernel_size = kernel_size, strides = 1, padding = "same", name = "{0}_Conv1D_3".format(name))(input_layer)

		block_layer = Add(name = "{0}_Identity_Add".format(name))([block_layer, conv_layer])
		return Activation("relu", name = "{0}_ReLU_2".format(name))(block_layer)

	def build_rgb_model(base_filter_size):		
		network = build_resnet_pooling_block(rgb_input, filters = base_filter_size, kernel_size = 7, name = "rgb_pooling_resnet_1")
		network = build_resnet_identity_block(network, filters = base_filter_size, kernel_size = 5, name = "rgb_identity_resnet_1")
		network = build_resnet_identity_block(network, filters = base_filter_size, kernel_size = 5, name = "rgb_identity_resnet_2")

		network = build_resnet_pooling_block(network, filters = 2 * base_filter_size, kernel_size = 7, name = "rgb_pooling_resnet_2")
		network = build_resnet_identity_block(network, filters = 2 * base_filter_size, kernel_size = 5, name = "rgb_identity_resnet_3")
		network = build_resnet_identity_block(network, filters = 2 * base_filter_size, kernel_size = 5, name = "rgb_identity_resnet_4")

		network = build_resnet_pooling_block(network, filters = 4 * base_filter_size, kernel_size = 7, name = "rgb_pooling_resnet_3")
		network = build_resnet_identity_block(network, filters = 4 * base_filter_size, kernel_size = 5, name = "rgb_identity_resnet_5")
		network = build_resnet_identity_block(network, filters = 4 * base_filter_size, kernel_size = 5, name = "rgb_identity_resnet_6")

		return GlobalAveragePooling1D(name = "gap_rgb")(network)

	def build_ppg_model(base_filter_size):
		network = build_resnet_pooling_block(ppg_input, filters = base_filter_size, kernel_size = 7, name = "ppg_pooling_resnet_1")
		network = build_resnet_identity_block(network, filters = base_filter_size, kernel_size = 5, name = "ppg_identity_resnet_1")
		network = build_resnet_identity_block(network, filters = base_filter_size, kernel_size = 5, name = "ppg_identity_resnet_2")

		network = build_resnet_pooling_block(network, filters = 2 * base_filter_size, kernel_size = 7, name = "ppg_pooling_resnet_2")
		network = build_resnet_identity_block(network, filters = 2 * base_filter_size, kernel_size = 5, name = "ppg_identity_resnet_3")
		network = build_resnet_identity_block(network, filters = 2 * base_filter_size, kernel_size = 5, name = "ppg_identity_resnet_4")

		network = build_resnet_pooling_block(network, filters = 4 * base_filter_size, kernel_size = 7, name = "ppg_pooling_resnet_3")
		network = build_resnet_identity_block(network, filters = 4 * base_filter_size, kernel_size = 5, name = "ppg_identity_resnet_5")
		network = build_resnet_identity_block(network, filters = 4 * base_filter_size, kernel_size = 5, name = "ppg_identity_resnet_6")
		
		return GlobalAveragePooling1D(name = "gap_ppg")(network)

	def build_fft_model(base_filter_size):
		network = Flatten(name = "fft_flatten")(fft_input)
		return Dense(base_filter_size, activation = "relu")(network)

	rgb_input = Input(shape = (frame_count, 3), name = "rgb_input")
	ppg_input = Input(shape = (frame_count, 2), name = "ppg_input")
	fft_input = Input(shape = (frame_count // 2, 2), name = "fft_input")
	
	build_functions = [build_rgb_model, build_ppg_model, build_fft_model]
	rgb_model, ppg_model, fft_model = [call(base_filter_size) for call in build_functions]

	embeddings = Concatenate(name = "merge_combined")([rgb_model, ppg_model, fft_model])
	combined = Dropout(rate = 0.5, name = "dropout_combined1")(embeddings)
	combined = Dense(2, activation = "softmax", name = "final_layer")(combined)

	combined = Model(inputs = [rgb_input, ppg_input, fft_input], outputs = combined)

	combined.compile(
		optimizer = Adam(lr = learning_rate),
		loss = "binary_crossentropy",
		metrics = ["accuracy", FPR, FNR]
	)
	combined.summary()
	return [rgb_input, ppg_input, fft_input], combined

import argparse

def build_parser():
	parser = argparse.ArgumentParser(description = "Código para treino das redes neurais utilizadas para detecção de ataques de apresentação.")
	parser.add_argument("data_folder", nargs = None, default = None, action = "store",
						help = "Localização da pasta do dataset (ou dos arquivos pré-processados). (Padrão: %(default)s)")
	parser.add_argument("--time", dest = "time", nargs = "?", default = 5, action = "store", type = int,
						help = "Tempo (em segundos) de captura ou análise. (Padrão: %(default)s segundos)")
	parser.add_argument("--process_data", dest = "process_data", action = "store_true", default = False,
						help = "Define se serão obtidas novas médias a partir do dataset informado. (Padrão: %(default)s)")
	return parser

if __name__ == "__main__":
	args = build_parser().parse_args()
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	frame_rate = 25

	if args.process_data is True:
		from utils.loaders import SiW_Loader
		SiW_Loader(args.data_folder).load_data(time = args.time)