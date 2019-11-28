import keras.backend as K
import numpy as np
import argparse
import cv2
import os


from sklearn.model_selection import train_test_split


from keras.layers import Dense, BatchNormalization, Activation, Flatten, Lambda
from keras.layers import Conv1D, Input, MaxPooling1D, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.callbacks import Callback
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


def APCER(y_true, y_pred):
	y_pred = np.argmax(y_pred, axis=1)
	count_fake = np.sum(1 - y_true)

	false_positives = np.all([y_true == 0, y_pred == 1], axis=0)
	false_positives = np.sum(false_positives)
	return false_positives / count_fake


def BPCER(y_true, y_pred):
	y_pred = np.argmax(y_pred, axis=1)
	count_real = np.sum(y_true)

	false_negatives = np.all([y_true == 1, y_pred == 0], axis=0)
	false_negatives = np.sum(false_negatives)
	return false_negatives / count_real


def ACER(apcer, bpcer):
	return (apcer + bpcer) / 2.0


def evaluate_on_data(y_true, y_pred):
	apcer = APCER(y_true, y_pred)
	bpcer = BPCER(y_true, y_pred)
	acer = ACER(apcer, bpcer)

	acc = sum(y_true == y_pred.argmax(axis=1)) / len(y_true)

	return acer, "APCER={:.5f}, BPCER={:.5f}, ACER={:.5f}, ACC={}".format(apcer,
																		  bpcer,
																		  acer,
																		  acc)


class EvaluationCallback(Callback):
	def __init__(self, destination=None, x=None, y=None, verbose=True):
		self.destination = destination
		self.best_acer = np.inf
		self.verbose = verbose
		self.epochs = 1

		self.x = x
		self.y = y

	def on_epoch_end(self, batch, logs):
		devel_x, devel_y = self.x, self.y
		pred_y = self.model.predict(devel_x)

		acer, evaluation = evaluate_on_data(devel_y, pred_y)

		print("(Epoch {}) {}.".format(self.epochs, evaluation))
		self.epochs += 1

		if self.best_acer > acer and self.destination is not None:
			if self.verbose:
				print('\n[ModelCheckpoint] ACER improved from {:.5f} to {:.5f}.'\
					  ' Storing best weights in {}\n'.format(self.best_acer,
															 acer,
															 self.destination))
			self.best_acer = acer

			self.model.save_weights(self.destination)
		elif self.best_acer < acer and (self.verbose):
			print('\n[ModelCheckpoint] ACER not improved from {:.5f}\n'.format(self.best_acer))


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

			# ppg_x = ppg_x[:, :, -1:] #REMOVER ISTO (TREINAR COM TODOS) TODO

			return p_x, ppg_x, p_y

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

		def solve_nan_positions(data):
			isnan = np.isnan(data)
			data[isnan == True] = 0.0
			return data

		train_rppg_x = solve_nan_positions(train_rppg_x)
		test_rppg_x = solve_nan_positions(test_rppg_x)

		train_x = solve_nan_positions(train_x)
		test_x = solve_nan_positions(test_x)

		train_x = train_x / 255.0
		test_x = test_x / 255.0

		shuffle(train_x, train_rppg_x, train_y)
		train_x, devel_x, train_rppg_x, devel_rppg_x, train_y, devel_y = train_test_split(train_x, train_rppg_x, train_y, test_size=0.2)
	elif args.prefix == 'oulu':
		def load_split(split):
			rppg = np.load('{0}/oulu_{1}_rppg_x.npy'.format(args.source, split))
			x = np.load('{0}/oulu_{1}_x.npy'.format(args.source, split))
			y = np.load('{0}/oulu_{1}_y.npy'.format(args.source, split))

			x = x / 255.0

			return x, rppg, y

		train_x, train_rppg_x, train_y = load_split('train')
		devel_x, devel_rppg_x, devel_y = load_split('devel')
		test_x, test_rppg_x, test_y = load_split('test')
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
	from model.structures import SimpleResnetRPPG
	from model.structures import SimpleResnetRGB
	from model.structures import LSTM_RPPG
	from model.structures import CNN_LSTM_RPPG

	architectures = [SimpleConvolutionalRGB,
					 SimpleConvolutionalRPPG,
					 DeepConvolutionalRGB,
					 DeepConvolutionalRPPG,
					 SimpleResnetRGB,
					 SimpleResnetRPPG,
					 CNN_LSTM_RPPG,
					 LSTM_RPPG]

	architectures = [SimpleConvolutionalRPPG,
					 DeepConvolutionalRPPG,
					 SimpleResnetRPPG,
					 CNN_LSTM_RPPG,
					 LSTM_RPPG]

	architectures = [DeepConvolutionalRPPG]

	batch_size = 16
	verbose = True
	epochs = 1000

	for arch in architectures:
		arch_name = arch.__name__
		model_dest = "./model/models/{}_{}.ckpt".format(args.prefix, arch_name)

		true_count = np.sum(train_y)
		fake_count = len(train_y) - true_count
		true_weight = fake_count / true_count
		if not arch.uses_rppg():
			arch_model = arch(shape=train_x.shape[1:], lr=1e-4, verbose=verbose)

			model_ckpt = EvaluationCallback(destination=model_dest, x=devel_x, y=devel_y)
			arch_model.fit(x=train_x, y=train_y,
									  epochs=epochs,
									  batch_size=batch_size,
									  callbacks=[model_ckpt],
									  class_weight={0:1, 1:true_weight})

			arch_model.get_model().load_weights(model_dest)
			y_pred = arch_model.get_model().predict(test_x)
			_, evaluation = evaluate_on_data(test_y, y_pred)

			with open('{}_evaluation.txt'.format(args.prefix), 'a') as file:
				file.write("{0} : {1}\n".format(arch_name, evaluation))
		else:
			arch_model = arch(shape=train_rppg_x.shape[1:], lr=1e-3, verbose=verbose)

			model_ckpt = EvaluationCallback(destination=model_dest, x=devel_rppg_x, y=devel_y)
			arch_model.fit(x=train_rppg_x, y=train_y,
										   epochs=epochs,
										   batch_size=batch_size,
										   callbacks=[model_ckpt],
										   class_weight={0:1, 1:true_weight})

			arch_model.get_model().load_weights(model_dest)

			arch_model.get_model().load_weights(model_dest)
			y_pred = arch_model.get_model().predict(test_rppg_x)
			_, evaluation = evaluate_on_data(test_y, y_pred)

			with open('{}_evaluation.txt'.format(args.prefix), 'a') as file:
				file.write("{0} : {1}\n".format(arch_name, evaluation))
