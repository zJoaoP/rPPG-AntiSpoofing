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

	def shuffle(a, b, c):
		rng_state = np.random.get_state()
		np.random.shuffle(a)
		np.random.set_state(rng_state)
		np.random.shuffle(b)
		np.random.set_state(rng_state)
		np.random.shuffle(c)

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

		shuffle(train_x, train_rppg_x, train_y)
	elif args.prefix == 'oulu':
		def load_split(split):
			rppg = np.load('{0}/oulu_{1}_rppg_x.npy'.format(args.source, split))
			x = np.load('{0}/oulu_{1}_x.npy'.format(args.source, split))
			y = np.load('{0}/oulu_{1}_y.npy'.format(args.source, split))

			y = to_categorical(y, 2)

			# def normalize(data):
			# 	return (data - data.mean()) / data.std()

			# rppg = normalize(rppg)
			# x = normalize(x)
			x = x / 255.0

			shuffle(x, rppg, y)

			return x, rppg, y

		train_x, train_rppg_x, train_y = load_split('train')
		devel_x, devel_rppg_x, devel_y = load_split('devel')
		test_x, test_rppg_x, test_y = load_split('test')

		# train_x = norm(train_x)
		# devel_x = norm(devel_x)
		# test_x = norm(test_x)

		# train_rppg_x = norm(train_rppg_x)
		# devel_rppg_x = norm(devel_rppg_x)
		# test_rppg_x = norm(test_rppg_x)

	elif args.prefix == "3DMAD":
		train_rppg_x = np.load("{0}/3DMAD_train_rppg_x.npy".format(args.source))
		train_x = np.load("{0}/3DMAD_train_x.npy".format(args.source))
		train_y = np.load("{0}/3DMAD_train_y.npy".format(args.source))

		test_rppg_x = np.load("{0}/3DMAD_test_rppg_x.npy".format(args.source))
		test_x = np.load("{0}/3DMAD_test_x.npy".format(args.source))
		test_y = np.load("{0}/3DMAD_test_y.npy".format(args.source))

		train_y = to_categorical(train_y, 2)
		test_y = to_categorical(test_y, 2)

	from model.structures import SimpleConvolutionalRPPG
	from model.structures import SimpleConvolutionalRGB
	from model.structures import DeepConvolutionalRPPG
	from model.structures import DeepConvolutionalRGB
	from model.structures import SimpleResnetRGB
	architectures = [SimpleConvolutionalRGB,
					 SimpleConvolutionalRPPG,
					 DeepConvolutionalRGB,
					 DeepConvolutionalRPPG]

	batch_size = 16
	verbose = False
	epochs = 1000

	eer = list()
	for arch in architectures:
		arch_model = arch(dimension=train_x.shape[1], lr=1e-4, verbose=verbose)
		arch_name = type(arch_model).__name__

		model_dest = "./model/models/{0}.ckpt".format(arch_name)
		model_ckpt = ModelCheckpoint(model_dest, monitor='val_ACER',
												 save_best_only=True,
												 mode='min',
												 verbose=verbose)

		early_stopping = EarlyStopping(monitor='val_ACER', mode='min', patience=300)
		callbacks = [model_ckpt, early_stopping]
		if not arch_model.uses_rppg():
			validation = None if devel_x is None else (devel_x, devel_y)
			arch_model.fit(x=train_x, y=train_y,
									  epochs=epochs,
									  batch_size=batch_size,
									  # validation_split=0.2,
									  validation_data=validation,
									  callbacks=callbacks)

			arch_model.get_model().load_weights(model_dest)
			pred = arch_model.get_model().predict(test_x)
			evaluation = arch_model.evaluate(test_x, test_y)
			print("{0} : {1}".format(arch_name, evaluation))
		else:
			validation = None if devel_x is None else ([devel_x, devel_rppg_x], devel_y)
			arch_model.fit(x=[train_x, train_rppg_x], y=train_y,
													  epochs=epochs,
													  batch_size=batch_size,
													  # validation_split=0.2,
													  validation_data=validation,
													  callbacks=callbacks)

			arch_model.get_model().load_weights(model_dest)

			evaluation = arch_model.evaluate([test_x, test_rppg_x], test_y)
			pred = arch_model.get_model().predict([test_x, test_rppg_x])
			print("{0} : {1}".format(arch_name, evaluation))

		# plot_model(model, to_file='./model/models/images/{0}.png'.format(arch_name))
		# pred = pred[:, 1]
		# from sklearn.metrics import roc_curve, auc, roc_auc_score
		# import matplotlib.pyplot as plt
		# fpr, tpr, threshold = roc_curve(np.argmax(train_y, axis=1), pred)
		# auc_roc = auc(fpr, tpr)

		# plt.title('Receiver Operating Characteristic')
		# plt.plot(fpr, tpr, 'b', label='AUC = {0:2f}'.format(auc_roc))
		# plt.legend(loc='lower right')
		# plt.plot([0, 1], [0, 1],'r--')
		# plt.xlim([0, 1])
		# plt.ylim([0, 1])
		# plt.ylabel('True Positive Rate')
		# plt.xlabel('False Positive Rate')
		# plt.show()

	# 	fnr = 1 - tpr
	# 	eer.append(fpr[np.nanargmin(np.absolute((fnr - fpr)))])
	# 	print("EER = %f" % (fpr[np.nanargmin(np.absolute((fnr - fpr)))]))

	# with open('3dmad_eer.txt', 'a+') as f:
	# 	f.write("{0}\n".format(eer))