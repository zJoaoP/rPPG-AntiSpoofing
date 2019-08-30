import keras.backend as K
import numpy as np
import argparse
import cv2
import os


from keras.layers import Dense, BatchNormalization, Activation, Flatten
from keras.layers import Conv1D, Input, MaxPooling1D, Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard
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
		y_true = K.cast(K.argmax(y_true, axis=1), dtype='float32')
		y_pred = K.cast(K.argmax(y_pred, axis=1), dtype='float32')
		count_fake = K.sum(1 - y_true)

		false_positives = K.all([K.equal(y_true, 0), K.equal(y_pred, 1)], axis=0)
		false_positives = K.sum(K.cast(false_positives, dtype='float32'))
		return K.switch(K.equal(count_fake, 0), 0.0, false_positives / count_fake)

	def BPCER(y_true, y_pred):
		y_true = K.cast(K.argmax(y_true, axis=1), dtype='float32')
		y_pred = K.cast(K.argmax(y_pred, axis=1), dtype='float32')
		count_real = K.sum(y_true)

		false_negatives = K.all([K.equal(y_true, 1), K.equal(y_pred, 0)], axis=0)
		false_negatives = K.sum(K.cast(false_negatives, dtype='float32'))
		return K.switch(K.equal(count_real, 0), 0.0, false_negatives / count_real)

	def ACER(y_true, y_pred):
		apcer = APCER(y_true, y_pred)
		bpcer = BPCER(y_true, y_pred)

		return (apcer + bpcer) / 2

	input_rppg = Input(shape=(input_shape[0], 1))
	input_rgb = Input(shape=input_shape)

	ppg_branch = Conv1D(64, kernel_size=5, strides=2, activation='linear', use_bias=False)(input_rppg)
	ppg_branch = BatchNormalization()(ppg_branch)
	ppg_branch = Activation('relu')(ppg_branch)

	rgb_branch = Conv1D(64, kernel_size=5, strides=2, activation='linear', use_bias=False)(input_rgb)
	rgb_branch = BatchNormalization()(rgb_branch)
	rgb_branch = Activation('relu')(rgb_branch)

	ppg_branch = Flatten()(ppg_branch)
	rgb_branch = Flatten()(rgb_branch)

	# ppg_branch = Flatten()(input_rppg)
	# rgb_branch = Flatten()(input_rgb)

	combined_branch = Concatenate()([rgb_branch, ppg_branch])
	combined_branch = Dense(2, activation='softmax')(combined_branch)

	model = Model([input_rgb, input_rppg], combined_branch)

	model.compile(
		optimizer=Adam(lr=learning_rate),
		loss="categorical_crossentropy",
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

		model.fit(x=[train_x, train_rppg_x], y=train_y,
											 epochs=200,
											 batch_size=16,
											 validation_split=0.25,
											 callbacks=[model_ckpt])

		model.load_weights('./models/model.ckpt')
		results_train = model.evaluate([train_x, train_rppg_x], train_y, batch_size=16)
		results_test = model.evaluate([test_x, test_rppg_x], test_y, batch_size=16)

		print(results_train)
		print(results_test)
		
		y_pred_raw = model.predict([test_x, test_rppg_x], verbose=0)
		def evaluate_by_threshold(x, y, threshold=0.0):
			y_true = np.argmax(y, axis=1)
			n_fake = np.sum(1 - y_true)
			n_real = np.sum(y_true)

			real_prob = y_pred_raw[:, 1]
			
			y_pred = np.argmax(y_pred_raw, axis=1)
			for i in range(len(y)):
				y_pred[i] = int(real_prob[i] >= threshold)

			apcer = np.sum(np.all([y_true == 0, y_pred == 1], axis=0)) / n_fake
			bpcer = np.sum(np.all([y_true == 1, y_pred == 0], axis=0)) / n_real
			acc = np.sum(y_pred == y_true, dtype=np.float64) / len(y_true)

			return apcer, bpcer, acc

		best_acer, best_th = 1.0, 0.0
		for th in np.arange(0.0, 1.01, 0.01):
			apcer, bpcer, acc = evaluate_by_threshold(test_x, test_y, th)
			acer = (apcer + bpcer) / 2.0
			print("%.2f : (%f, %f, %f, %f)" % (th, acc, apcer, bpcer, acer))

			if acer < best_acer:
				best_acer = acer
				best_th = th

		print(best_th)
		apcer, bpcer, _ = evaluate_by_threshold(test_x, test_y, best_th)
		print(apcer)
		print(bpcer)