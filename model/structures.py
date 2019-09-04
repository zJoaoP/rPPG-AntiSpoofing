import keras.backend as K
import tensorflow as tf


from keras.layers import Input, Flatten, Conv1D, GlobalAveragePooling1D
from keras.layers import Concatenate, Lambda, Dense
from model.metrics import APCER, BPCER, ACER
from keras.optimizers import Adam
from keras.models import Model


class GenericArchitecture:
	def fit(self, **kwargs):
		kwargs['verbose'] = self.verbose
		self.model.fit(**kwargs)

	def evaluate(self, x, y):
		arch_name = type(self).__name__
		evaluation = self.model.evaluate(x, y, verbose=self.verbose)
		results = dict(zip(self.model.metrics_names, evaluation))
		str_representation = "{0} : {1}".format(arch_name, results)
		return str_representation

	@staticmethod
	def triplet_loss(margin=1.0):
			triplet_semihard_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss
			def __triplet_loss(y_true, y_pred):
				from keras.losses import categorical_crossentropy
				triplet_contribution = triplet_semihard_loss(K.argmax(y_true, axis=1), y_pred)
				classification_contribution = categorical_crossentropy(y_true, y_pred)
				return triplet_contribution + classification_contribution

			return __triplet_loss

	def get_model(self):
		return self.model


class FlatRGB(GenericArchitecture):
	def __init__(self, frame_rate, time=15, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_shape=(frame_rate * time, 3))

	def build_model(self, input_shape):
		input_layer = Input(shape=input_shape)
		x = Flatten()(input_layer)
		x = Dense(2, activation='softmax')(x)

		model = Model(input_layer, x)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='categorical_crossentropy',
			metrics=['accuracy', APCER, BPCER, ACER]
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return False


class SimpleConvolutionalRGB(GenericArchitecture):
	def __init__(self, frame_rate, time=15, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_shape=(frame_rate * time, 3))

	def build_model(self, input_shape):
		input_layer = Input(shape=input_shape)
		x = Conv1D(64, kernel_size=5, strides=1, activation='relu')(input_layer)
		x = GlobalAveragePooling1D()(x)
		x = Dense(2, activation='softmax')(x)

		model = Model(input_layer, x)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='categorical_crossentropy',
			metrics=['accuracy', APCER, BPCER, ACER]
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return False


class DeepConvolutionalRGB(GenericArchitecture):
	def __init__(self, frame_rate, time=15, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_shape=(frame_rate * time, 3))

	def build_model(self, input_shape):
		input_layer = Input(shape=input_shape)
		x = Conv1D(64, kernel_size=5, strides=2, activation='relu')(input_layer)
		x = Conv1D(128, kernel_size=5, strides=2, activation='relu')(x)
		x = GlobalAveragePooling1D()(x)
		x = Dense(2, activation='softmax')(x)

		model = Model(input_layer, x)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='categorical_crossentropy',
			metrics=['accuracy', APCER, BPCER, ACER]
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return False


class FlatRPPG(GenericArchitecture):
	def __init__(self, frame_rate, time=15, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_dim=frame_rate * time)

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
			loss='categorical_crossentropy',
			metrics=['accuracy', APCER, BPCER, ACER]
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return True

class SimpleConvolutionalRPPG(GenericArchitecture):
	def __init__(self, frame_rate, time=15, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_dim=frame_rate * time)

	def build_model(self, input_dim):
		input_rgb = Input(shape=(input_dim, 3), name='input_rgb')
		input_ppg = Input(shape=(input_dim, 1), name='input_ppg')
		
		rgb_branch = Conv1D(64, kernel_size=5,
								strides=1,
								activation='relu')(input_rgb)
		
		ppg_branch = Conv1D(64, kernel_size=5,
								strides=1,
								activation='relu')(input_ppg)

		rgb_branch = GlobalAveragePooling1D()(rgb_branch)
		ppg_branch = GlobalAveragePooling1D()(ppg_branch)

		combined_branch = Concatenate()([rgb_branch, ppg_branch])
		combined_branch = Dense(2, activation='softmax')(combined_branch)

		model = Model([input_rgb, input_ppg], combined_branch)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='categorical_crossentropy',
			metrics=['accuracy', APCER, BPCER, ACER]
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return True


class DeepConvolutionalRPPG(GenericArchitecture):
	def __init__(self, frame_rate, time=15, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_dim=frame_rate * time)

	def build_model(self, input_dim):
		input_rgb = Input(shape=(input_dim, 3), name='input_rgb')
		input_ppg = Input(shape=(input_dim, 1), name='input_ppg')
		
		rgb_branch = Conv1D(64, kernel_size=5,
								strides=2,
								activation='relu')(input_rgb)

		rgb_branch = Conv1D(128, kernel_size=5,
								strides=2,
								activation='relu')(rgb_branch)
		
		ppg_branch = Conv1D(64, kernel_size=5,
								strides=2,
								activation='relu')(input_ppg)

		ppg_branch = Conv1D(128, kernel_size=5,
								strides=2,
								activation='relu')(ppg_branch)

		rgb_branch = GlobalAveragePooling1D()(rgb_branch)
		ppg_branch = GlobalAveragePooling1D()(ppg_branch)

		combined_branch = Concatenate()([rgb_branch, ppg_branch])
		combined_branch = Dense(2, activation='softmax')(combined_branch)

		model = Model([input_rgb, input_ppg], combined_branch)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss='categorical_crossentropy',
			metrics=['accuracy', APCER, BPCER, ACER]
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return True


class TripletRGB(GenericArchitecture):
	def __init__(self, frame_rate, time=15, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_shape=(frame_rate * time, 3))

	def build_model(self, input_shape):
		input_layer = Input(shape=input_shape)
		embeddings = Conv1D(64, kernel_size=5, strides=2, activation='relu')(input_layer)
		embeddings = Conv1D(128, kernel_size=5, strides=2, activation='relu')(embeddings)
		embeddings = GlobalAveragePooling1D()(embeddings)
		embeddings = Lambda(lambda x : K.l2_normalize(x, axis=1))(embeddings)
		classification = Dense(2, activation='softmax')(embeddings)

		model = Model(input_layer, classification)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss=GenericArchitecture.triplet_loss(margin=1.0),
			metrics=['accuracy', APCER, BPCER, ACER]
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return False


class TripletRPPG(GenericArchitecture):
	def __init__(self, frame_rate, time=15, lr=1e-4, verbose=False):
		self.learning_rate = lr
		self.verbose = verbose

		self.model = self.build_model(input_dim=frame_rate * time)

	def build_model(self, input_dim):
		input_rgb = Input(shape=(input_dim, 3), name='input_rgb')
		input_ppg = Input(shape=(input_dim, 1), name='input_ppg')
		
		rgb_branch = Conv1D(64, kernel_size=5,
								strides=2,
								activation='relu')(input_rgb)

		rgb_branch = Conv1D(128, kernel_size=5,
								strides=2,
								activation='relu')(rgb_branch)
		
		ppg_branch = Conv1D(64, kernel_size=5,
								strides=2,
								activation='relu')(input_ppg)

		ppg_branch = Conv1D(128, kernel_size=5,
								strides=2,
								activation='relu')(ppg_branch)

		rgb_branch = GlobalAveragePooling1D()(rgb_branch)
		ppg_branch = GlobalAveragePooling1D()(ppg_branch)

		combined_branch = Concatenate()([rgb_branch, ppg_branch])
		classification = Dense(2, activation='softmax')(combined_branch)

		model = Model([input_rgb, input_ppg], classification)
		model.compile(
			optimizer=Adam(lr=self.learning_rate),
			loss=GenericArchitecture.triplet_loss(margin=1.0),
			metrics=['accuracy', APCER, BPCER, ACER]
		)
		if self.verbose:
			model.summary()

		return model

	def uses_rppg(self):
		return True
