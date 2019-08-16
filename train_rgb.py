import keras.backend as K
import numpy as np
import argparse
import cv2
import os


from keras.layers import Dense, BatchNormalization, Activation, Flatten, LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, Input, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam

def build_model(input_shape, learning_rate=1e-3):
	def FAR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_true, y_pred], axis=0))

	def FRR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_pred, y_true], axis=0))

	def HTER(y_true, y_pred):
		far = FAR(y_true, y_pred)
		frr = FRR(y_true, y_pred)
		return (far + frr) / 2.0

	def APCER(y_true, y_pred):
		zero = K.variable(0)
		one = K.variable(1)
		y_true = K.cast(K.argmax(y_true, axis=1), dtype='float32')
		y_pred = K.cast(K.argmax(y_pred, axis=1), dtype='float32')
		
		true_classifications = K.equal(y_true, y_pred)
		zero_classifications = K.equal(y_pred, zero)
		true_zeros = K.all([true_classifications, zero_classifications], axis=0)
		
		attack_count = K.sum(K.cast(K.equal(y_true, zero), dtype='float32'))
		class_score = K.sum(K.cast(true_zeros, 'float32'))
		return K.switch(K.equal(attack_count, zero),
						zero,
						one - (class_score / attack_count))

	def BPCER(y_true, y_pred):
		zero = K.variable(0)
		one = K.variable(1)
		y_true = K.cast(K.argmax(y_true, axis=1), dtype='float32')
		y_pred = K.cast(K.argmax(y_pred, axis=1), dtype='float32')
		
		true_classifications = K.equal(y_true, y_pred)
		one_classifications = K.equal(y_pred, one)
		true_ones = K.all([true_classifications, one_classifications], axis=0)
		
		real_count = K.sum(K.cast(K.equal(y_true, one), dtype='float32'))
		class_score = K.sum(K.cast(true_ones, 'float32'))
		return K.switch(K.equal(real_count, zero),
						zero,
						one - (class_score / real_count))

	def ACER(y_true, y_pred):
		apcer = APCER(y_true, y_pred)
		bpcer = BPCER(y_true, y_pred)

		return (apcer + bpcer) / 2

	model = Sequential()

	model.add(Conv1D(128, kernel_size=7, strides=2,
							activation='linear',
							use_bias=False,
							input_shape=input_shape))

	model.add(BatchNormalization())
	model.add(Activation('relu'))
	
	model.add(Conv1D(64, kernel_size=5, strides=2,
							activation='linear',
							use_bias=False))

	model.add(BatchNormalization())
	model.add(Activation('relu'))
	
	model.add(Flatten())
	
	model.add(Dense(2, activation='softmax'))

	# def triplet_loss(margin=1.0):
	# 	import tensorflow as tf
	# 	def triplet_semihard(y_true, y_pred):
	# 		return tf.contrib.losses.metric_learning.triplet_semihard_loss(K.argmax(y_true, axis=1), y_pred, margin)

	# 	return triplet_semihard

	model.compile(
		optimizer=Adam(lr=learning_rate),
		loss="binary_crossentropy",
		metrics=['accuracy', APCER, BPCER, ACER]
	)
	model.summary()
	return model


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
	if args.prefix == 'rad':
		def load_partition(name, flip=True):
			p_x = np.load("{0}/{1}_{2}_x.npy".format(args.source,
														args.prefix,
														name))
			p_y = np.load("{0}/{1}_{2}_y.npy".format(args.source,
														args.prefix,
														name))
			
			if flip:
				p_x = np.append(p_x, np.flip(p_x, axis=1), axis = 0)
				p_y = np.append(p_y, p_y)

			p_x = p_x / 255.0
			return p_x, to_categorical(p_y, 2)

		train_x, train_y = load_partition('train')
		devel_x, devel_y = load_partition('devel', flip=False)
		test_x, test_y = load_partition('test', flip=False)

		model = build_model(train_x[0].shape, learning_rate=1e-3)
		model_ckpt = ModelCheckpoint('./models/model.ckpt', monitor='val_acc',
															save_best_only=True,
															verbose=1)

		model.fit(x=train_x, y=train_y, epochs=100, batch_size=16,
										validation_data=(devel_x, devel_y),
										callbacks=[model_ckpt])

	elif args.prefix == 'siw':
		train_x = np.load('{0}/siw_train_x.npy'.format(args.source))
		train_y = np.load('{0}/siw_train_y.npy'.format(args.source))

		test_x = np.load('{0}/siw_test_x.npy'.format(args.source))
		test_y = np.load('{0}/siw_test_y.npy'.format(args.source))
		
		train_x = np.append(train_x, np.flip(train_x, axis=1), axis=0)
		train_y = np.append(train_y, train_y)

		train_y = to_categorical(train_y, 2)
		test_y = to_categorical(test_y, 2)

		train_x = train_x / 255.0
		test_x = test_x / 255.0

		train_x = np.nan_to_num(train_x)
		test_x = np.nan_to_num(test_x)

		def shuffle(a, b):
			rng_state = np.random.get_state()
			np.random.shuffle(a)
			np.random.set_state(rng_state)
			np.random.shuffle(b)


		shuffle(train_x, train_y)

		model = build_model(train_x[0].shape, learning_rate=1e-4)

		model_ckpt = ModelCheckpoint('./models/model.ckpt', monitor='val_acc',
															save_best_only=True,
															verbose=1)

		model.fit(x=train_x, y=train_y, epochs=200,
										batch_size=16,
										validation_split=0.30,
										callbacks=[model_ckpt])

		model.load_weights('./models/model.ckpt')
		results = model.evaluate(test_x, test_y)
		print(results)