import keras.backend as K
import numpy as np
import argparse
import cv2
import os


from keras.layers import Dense, Flatten, BatchNormalization, Activation
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam

def build_model(input_shape, learning_rate = 1e-3):
	def FPR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_true, y_pred], axis = 0))

	def FNR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_pred, y_true], axis = 0))

	model = Sequential()

	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(2, activation='sigmoid'))

	model.compile(
		optimizer=Adam(lr=learning_rate),
		loss='binary_crossentropy',
		metrics=['accuracy', FPR, FNR]
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

	parser.add_argument("--time", default = 5, type = int,
							help = "Tempo (em segundos) de captura ou análise. \
									(Padrão: %(default)s segundos)")

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
		devel_x, devel_y = load_partition('devel')
		test_x, test_y = load_partition('test', flip=False)

		model = build_model(train_x[0].shape)
		model_ckpt = ModelCheckpoint('./models/model.ckpt', monitor='val_acc',
															save_best_only=True,
															verbose=1)

		model.fit(x=train_x, y=train_y, epochs=500, batch_size=16,
										validation_data=(devel_x, devel_y),
										callbacks = [model_ckpt])

		model.load_weights('./models/model.ckpt')
		result = model.evaluate(x=test_x, y=test_y)
		print(result)
