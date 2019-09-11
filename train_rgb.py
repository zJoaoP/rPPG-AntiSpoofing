import keras.backend as K
import numpy as np
import argparse
import cv2
import os


from keras.layers import Dense, BatchNormalization, Activation, Flatten, Lambda
from keras.layers import Conv1D, Input, MaxPooling1D, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.optimizers import Adam


def build_parser():
	parser = argparse.ArgumentParser(description = "Código para treino das \
													redes neurais utilizadas \
													para detecção de ataques \
													de apresentação.")
	parser.add_argument("source",
							help = "Localização dos arquivos pré-processdos. \
									(Padrão: %(default)s)")

	parser.add_argument("--prefix", default = 'rad', type = str,
							help = "Prefixo do dataset a ser utilizado. \
									(Padrão: %(default)s)")
	return parser.parse_args()


if __name__ == "__main__":
	args = build_parser()
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	train_x, train_rppg_x, train_y = [None] * 3
	devel_x, devel_rppg_x, devel_y = [None] * 3
	test_x, test_rppg_x, test_y = [None] * 3
	if args.prefix == 'rad':
		def load_partition(name, flip=False):
			p_x = np.load("{0}/{1}_{2}_x.npy".format(args.source,
														args.prefix,
														name))
			p_y = np.load("{0}/{1}_{2}_y.npy".format(args.source,
														args.prefix,
														name))
			ppg_x = np.load("{0}/{1}_{2}_ppg.npy".format(args.source,
														args.prefix,
														name))
			
			if flip:
				p_x = np.append(p_x, np.flip(p_x, axis=1), axis = 0)
				p_y = np.append(p_y, p_y)

			p_x = p_x / 255.0
			return p_x, ppg_x, to_categorical(p_y, 2)

		train_x, train_rppg_x, train_y = load_partition('train')
		devel_x, devel_rppg_x, devel_y = load_partition('devel')
		test_x, test_rppg_x, test_y = load_partition('test')
	elif args.prefix == 'siw':
		train_rppg_x = np.load('{0}/siw_train_rppg_x.npy'.format(args.source))
		train_x = np.load('{0}/siw_train_x.npy'.format(args.source))
		train_y = np.load('{0}/siw_train_y.npy'.format(args.source))

		test_rppg_x = np.load('{0}/siw_test_rppg_x.npy'.format(args.source))
		test_x = np.load('{0}/siw_test_x.npy'.format(args.source))
		test_y = np.load('{0}/siw_test_y.npy'.format(args.source))

		train_y = to_categorical(train_y, 2)
		test_y = to_categorical(test_y, 2)

		train_x = train_x / 255.0
		test_x = test_x / 255.0

		def solve_nan_positions(data):
			isnan = np.isnan(data)
			data[isnan == True] = 0.0
			return data

		train_x = solve_nan_positions(train_x)
		test_x = solve_nan_positions(test_x)

		def shuffle(a, b, c):
			rng_state = np.random.get_state()
			np.random.shuffle(a)
			np.random.set_state(rng_state)
			np.random.shuffle(b)
			np.random.set_state(rng_state)
			np.random.shuffle(c)


		shuffle(train_x, train_rppg_x, train_y)
	elif args.prefix == 'oulu':
		def load_split(split):
			rppg = np.load('{0}/oulu_{1}_rppg_x.npy'.format(args.source, split))
			x = np.load('{0}/oulu_{1}_x.npy'.format(args.source, split))
			y = np.load('{0}/oulu_{1}_y.npy'.format(args.source, split))

			y = to_categorical(y, 2)

			def normalize(data):
				return (data - data.mean()) / data.std()

			rppg = normalize(rppg)
			x = normalize(x)

			return x, rppg, y

		train_x, train_rppg_x, train_y = load_split('train')
		devel_x, devel_rppg_x, devel_y = load_split('devel')
		test_x, test_rppg_x, test_y = load_split('test')

	from model.structures import SimpleConvolutionalRPPG
	from model.structures import SimpleConvolutionalRGB
	from model.structures import DeepConvolutionalRPPG
	from model.structures import DeepConvolutionalRGB
	from model.structures import TripletRPPG
	from model.structures import TripletRGB
	from model.structures import FlatRPPG
	from model.structures import FlatRGB
	architectures = [TripletRGB,
					 TripletRPPG,
					 SimpleConvolutionalRGB,
					 SimpleConvolutionalRPPG,
					 DeepConvolutionalRGB,
					 DeepConvolutionalRPPG,
					 FlatRGB,
					 FlatRPPG]

	batch_size = 16
	verbose = False
	epochs = 50000

	for arch in architectures:
		arch_model = arch(dimension=train_x.shape[1], lr=1e-4, verbose=verbose)
		arch_name = type(arch_model).__name__

		model_dest = "./model/models/{0}.ckpt".format(arch_name)
		model_ckpt = ModelCheckpoint(model_dest, monitor='val_ACER',
												 save_best_only=True,
												 verbose=verbose)
		early_stopping = EarlyStopping(monitor='val_ACER', patience=1000)
		callbacks = [model_ckpt, early_stopping]
		if not arch_model.uses_rppg():
			validation = None if train_x is None else (devel_x, devel_y)
			arch_model.fit(x=train_x, y=train_y,
									  epochs=epochs,
									  batch_size=batch_size,
									  validation_data=validation,
									  callbacks=callbacks)

			print(arch_model.evaluate(test_x, test_y))
		else:
			validation = None if train_x is None else ([devel_x, devel_rppg_x], devel_y)
			arch_model.fit(x=[train_x, train_rppg_x], y=train_y,
													  epochs=epochs,
													  batch_size=batch_size,
													  validation_data=validation,
													  callbacks=callbacks)

			print(arch_model.evaluate([test_x, test_rppg_x], test_y))

		model = arch_model.get_model()
		plot_model(model, to_file='./model/models/images/{0}.png'.format(arch_name))