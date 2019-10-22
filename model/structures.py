import keras.backend as K
import tensorflow as tf


from keras.layers import Input, Flatten, Conv1D, GlobalAveragePooling1D
from keras.layers import BatchNormalization, Activation, MaxPooling1D
from model.metrics import APCER, BPCER, ACER
from keras.layers import Concatenate, Lambda, Dense
from keras.optimizers import Adam
from keras.models import Model


class GenericArchitecture:
	def __init__(self, dimension, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_shape=(dimension, 3))

	def fit(self, **kwargs):
		kwargs['verbose'] = self.verbose
		self.model.fit(**kwargs)

	def evaluate(self, x, y):
		evaluation = self.model.evaluate(x, y, verbose=self.verbose)
		return dict(zip(self.model.metrics_names, evaluation))

	def get_model(self):
		return self.model


class FlatRGB(GenericArchitecture):
	def build_model(self, input_shape):
		input_layer = Input(shape=input_shape)
		x = Flatten()(input_layer)
		x = Dense(2, activation='softmax')(x)

		model = Model(input_layer, x)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='sparse_categorical_crossentropy',
			metrics=['acc']
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return False


class SimpleConvolutionalRGB(GenericArchitecture):
	def build_model(self, input_shape):
		input_layer = Input(shape=input_shape)
		
		x = Conv1D(64, kernel_size=5, strides=1, activation='linear', use_bias=False)(input_layer)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling1D(pool_size=3, strides=2)(x)

		x = GlobalAveragePooling1D()(x)

		x = Dense(2, activation='softmax')(x)

		model = Model(input_layer, x)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='sparse_categorical_crossentropy',
			metrics=['acc']
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return False

class SimpleResnetRGB(GenericArchitecture):
	def build_model(self, input_shape):
		from keras.layers import Add

		input_layer = Input(shape=input_shape)

		def resnet_identity_block(input_layer, filters):
			x = Conv1D(filters, kernel_size=5, strides=1, padding='same', activation='relu')(input_layer)
			x = Conv1D(filters, kernel_size=5, strides=1, padding='same', activation='linear')(x)
			x = Add()([x, input_layer])
			return Activation('relu')(x)

		def resnet_residual_block(input_layer, filters):
			x = Conv1D(filters, kernel_size=5, strides=2, activation='relu')(input_layer)
			x = Conv1D(filters, kernel_size=5, strides=1, padding='same', activation='linear')(x)
		
			input_layer = Conv1D(filters, kernel_size=5, strides=2, activation='linear')(input_layer)
			
			x = Add()([x, input_layer])
			return Activation('relu')(x)
		
		x = Conv1D(64, kernel_size=5, strides=2, activation='relu')(input_layer)

		for i in range(3):
			x = resnet_identity_block(x, 64)
		
		x = resnet_residual_block(x, 128)
		for i in range(3):
			x = resnet_identity_block(x, 128)

		x = resnet_residual_block(x, 256)
		for i in range(5):
			x = resnet_identity_block(x, 256)

		x = resnet_residual_block(x, 512)
		for i in range(2):
			x = resnet_identity_block(x, 512)

		x = GlobalAveragePooling1D()(x)
		x = Dense(2, activation='softmax')(x)

		model = Model(input_layer, x)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='sparse_categorical_crossentropy',
			metrics=['acc']
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return False


class DeepConvolutionalRGB(GenericArchitecture):
	def build_model(self, input_shape):
		input_layer = Input(shape=input_shape)
		x = Conv1D(64, kernel_size=5, strides=1, activation='linear', use_bias=False)(input_layer)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling1D(pool_size=3, strides=2)(x)

		x = Conv1D(128, kernel_size=5, strides=1, activation='linear', use_bias=False)(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling1D(pool_size=3, strides=2)(x)

		x = GlobalAveragePooling1D()(x)

		x = Dense(2, activation='softmax')(x)

		model = Model(input_layer, x)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='sparse_categorical_crossentropy',
			metrics=['acc']
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return False


class FlatRPPG(GenericArchitecture):
	def __init__(self, dimension, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_dim=dimension)

	def build_model(self, input_dim):
		input_rgb = Input(shape=(input_dim, 3), name='input_rgb')
		input_ppg = Input(shape=(input_dim, 1), name='input_ppg')
		
		rgb_branch = Flatten()(input_rgb)
		ppg_branch = Flatten()(input_ppg)

		combined_branch = Concatenate()([rgb_branch, ppg_branch])
		combined_branch = Dense(2, activation='softmax')(combined_branch)

		model = Model([input_rgb, input_ppg], combined_branch)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='sparse_categorical_crossentropy',
			metrics=['acc']
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return True

class SimpleConvolutionalRPPG(GenericArchitecture):
	def __init__(self, dimension, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_dim=dimension)

	def build_model(self, input_dim):
		input_rgb = Input(shape=(input_dim, 3), name='input_rgb')
		input_ppg = Input(shape=(input_dim, 1), name='input_ppg')
		
		rgb_branch = Conv1D(64, kernel_size=5,
								strides=1,
								activation='linear',
								use_bias=False)(input_rgb)

		rgb_branch = BatchNormalization()(rgb_branch)
		rgb_branch = Activation('relu')(rgb_branch)
		
		ppg_branch = Conv1D(64, kernel_size=5,
								strides=1,
								activation='linear',
								use_bias=False)(input_ppg)

		ppg_branch = BatchNormalization()(ppg_branch)
		ppg_branch = Activation('relu')(ppg_branch)

		rgb_branch = GlobalAveragePooling1D()(rgb_branch)
		ppg_branch = GlobalAveragePooling1D()(ppg_branch)

		combined_branch = Concatenate()([rgb_branch, ppg_branch])
		combined_branch = Dense(2, activation='softmax')(combined_branch)

		model = Model([input_rgb, input_ppg], combined_branch)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='sparse_categorical_crossentropy',
			metrics=['acc']
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return True


class DeepConvolutionalRPPG(GenericArchitecture):
	def __init__(self, dimension, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_dim=dimension)

	def build_model(self, input_dim):
		input_rgb = Input(shape=(input_dim, 3), name='input_rgb')
		input_ppg = Input(shape=(input_dim, 1), name='input_ppg')

		def ConvWithBN(input_layer, filters):
			x = Conv1D(filters, kernel_size=5, strides=2, activation='linear', use_bias=False)(input_layer)
			x = BatchNormalization()(x)
			return Activation('relu')(x)

		rgb_branch = ConvWithBN(input_rgb, 64)
		rgb_branch = ConvWithBN(rgb_branch, 128)

		ppg_branch = ConvWithBN(input_ppg, 64)
		ppg_branch = ConvWithBN(ppg_branch, 128)
		
		rgb_branch = GlobalAveragePooling1D()(rgb_branch)
		ppg_branch = GlobalAveragePooling1D()(ppg_branch)

		combined_branch = Concatenate()([rgb_branch, ppg_branch])
		combined_branch = Dense(2, activation='softmax')(combined_branch)

		model = Model([input_rgb, input_ppg], combined_branch)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='sparse_categorical_crossentropy',
			metrics=['acc']
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return True


class TripletRGB(GenericArchitecture):
	def build_model(self, input_shape):
		input_layer = Input(shape=input_shape)
		embeddings = Conv1D(64, kernel_size=5, strides=2, activation='linear', use_bias=False)(input_layer)
		embeddings = BatchNormalization()(embeddings)
		embeddings = Activation('relu')(embeddings)

		embeddings = Conv1D(128, kernel_size=5, strides=2, activation='linear', use_bias=False)(embeddings)
		embeddings = BatchNormalization()(embeddings)
		embeddings = Activation('relu')(embeddings)

		embeddings = GlobalAveragePooling1D()(embeddings)
		embeddings = Lambda(lambda x : K.l2_normalize(x, axis=1))(embeddings)
		classification = Dense(2, activation='softmax')(embeddings)

		def triplet_loss(margin=1.0):
			triplet_semihard_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss
			def __triplet_loss(y_true, y_pred):
				from keras.losses import sparse_categorical_crossentropy
				triplet_contribution = triplet_semihard_loss(K.argmax(y_true, axis=1), embeddings)
				classification_contribution = sparse_categorical_crossentropy(y_true, y_pred)
				return triplet_contribution + classification_contribution

			return __triplet_loss

		model = Model(input_layer, classification)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss=triplet_loss(margin=1.0),
			metrics=['acc']
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return False


class TripletRPPG(GenericArchitecture):
	def __init__(self, dimension, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_dim=dimension)

	def build_model(self, input_dim):
		input_rgb = Input(shape=(input_dim, 3), name='input_rgb')
		input_ppg = Input(shape=(input_dim, 1), name='input_ppg')
		
		def ConvWithBN(input_layer, filters):
			x = Conv1D(filters, kernel_size=5, strides=1, activation='linear', use_bias=False)(input_layer)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			return MaxPooling1D(pool_size=3, strides=2)(x)

		rgb_branch = ConvWithBN(input_rgb, 64)
		rgb_branch = ConvWithBN(rgb_branch, 128)

		ppg_branch = ConvWithBN(input_ppg, 64)
		ppg_branch = ConvWithBN(ppg_branch, 128)

		rgb_branch = GlobalAveragePooling1D()(rgb_branch)
		ppg_branch = GlobalAveragePooling1D()(ppg_branch)

		combined_branch = Concatenate()([rgb_branch, ppg_branch])
		embeddings = Lambda(lambda x : K.l2_normalize(x, axis=1))(combined_branch)

		classification = Dense(2, activation='softmax')(embeddings)
		
		def triplet_loss(margin=1.0):
			triplet_semihard_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss
			def __triplet_loss(y_true, y_pred):
				from keras.losses import sparse_categorical_crossentropy
				triplet_contribution = triplet_semihard_loss(K.argmax(y_true, axis=1), embeddings)
				classification_contribution = sparse_categorical_crossentropy(y_true, y_pred)
				return triplet_contribution + classification_contribution

			return __triplet_loss

		model = Model([input_rgb, input_ppg], classification)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss=triplet_loss(margin=1.0),
			metrics=['acc']
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return True
