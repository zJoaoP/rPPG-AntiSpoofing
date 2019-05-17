'''
	# Objetivo: Este arquivo possui (como objetivo principal) o treino de uma rede neural que recebe como entrada as features extraídas
	pelo algoritmo proposto por Gerard De Haan (CHROM) e exportará True (Em caso de ataque de apresentação) ou False (Caso contrário).

	# Etapas:
		1. Carregar os dados. (OK)
			1.1 - Definir classe para carregamento de dados. (Tentar desenvolver de modo que seja possível escolher quais arquivos serão carregados)
			1.2 - Escrever código simples para 'argparse'.
			1.3 - De que modo a estratégia será passada para a rede? Qual a saída? Será sempre um vetor com o tamanho da quantidade de frames?
		-------------------------------------------------------------------------------------
		2. Montar a rede. (Keras ou Tensorflow?)
			2.1 - Aprender a verificar se alguém está utilizando alguma gpu.
			2.2 - Testar antes na máquina local.
			2.3 - Criar um ambiente virtual (venv).

			2.4 - Devo usar uma FCN ou uma RNN?
		
		3. Treinar a rede.
'''

from utils.landmarks import LandmarkPredictor
from utils.extractor import CheeksAndNose

import numpy as np
import cv2
import os

class VideoLoader:
	def __init__(self, time = 10, roi_extractor = CheeksAndNose()):
		self.roi_extractor = roi_extractor
		self.time = time

	def is_valid_video(self, filename):
		valid_extensions = ["avi", "mp4", "mov"]
		extension = filename.split('.')[-1]
		return extension in valid_extensions

	def get_video_info(self, filename):
		if not self.is_valid_video(filename):
			return -1, -1
		else:
			stream = cv2.VideoCapture(filename)
			frame_count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
			frame_rate = int(stream.get(cv2.CAP_PROP_FPS))
			stream.release()

			return frame_rate, frame_count

	def load_video(self, filename):
		if self.is_valid_video(filename):
			stream = cv2.VideoCapture(filename)
			face_predictor = LandmarkPredictor()
			frame_rate, current_frame = int(stream.get(cv2.CAP_PROP_FPS)), 0
			means = []

			while True:
				success, frame = stream.read()
				if not success:
					break

				face_rect = face_predictor.detect_face(image = frame)
				if face_rect is not None:
					face_left, face_right, face_top, face_bottom = face_rect.left(), face_rect.right(), face_rect.top(), face_rect.bottom()
					cut_face = frame[face_top : face_bottom, face_left : face_right, :]

					landmarks = face_predictor.detect_landmarks(image = cut_face)
					roi = self.roi_extractor.extract_roi(cut_face, landmarks)
					
					if current_frame % 60 == 0 and current_frame > 0:
						print("[VideoLoader] {0} frames extracted from '{1}'.".format(current_frame, filename))
					
					roi = roi.astype(np.float32)
					roi[roi < 5.0] = np.nan

					r = np.nanmean(roi[:, :, 2])
					g = np.nanmean(roi[:, :, 1])
					b = np.nanmean(roi[:, :, 0])

					means += [[r, g, b]]

					current_frame += 1
				else:
					print("[VideoLoader] Skipping '{0}'. Reason: Failure on Acquisition! (It's a folder??)".format(filename))
					return None

			return means
		else:
			print("[VideoLoader] Skipping '{0}'. Reason: Invalid Video! (It's a folder??)".format(filename))
			return None

class DataLoader:
	def __init__(self, folder, file_list = None, time = 15, num_channels = 3, roi_extractor = CheeksAndNose()):
		self.roi_extractor = roi_extractor
		self.num_channels = num_channels
		self.time = time

		self.file_list = file_list
		self.folder = folder

		self.loader = VideoLoader(time = time, roi_extractor = roi_extractor)

	def split_raw_data(self, raw_data, split_size, stride = 1):
		data, i = None, 0
		while i + split_size - 1 < len(raw_data):
			data = np.array([raw_data[i : i + split_size, :]]) if data is None else np.append(data, np.array([raw_data[i : i + split_size, :]]), axis = 0)
			i += stride

		return data

	def post_process(self, data, frame_rate = 25):
		min_data_size = min([len(data[i]) for i in range(len(data))])
		min_data_size = min(min_data_size, self.time * frame_rate)
		final_data = None

		for i in range(len(data)):
			sub_videos = self.split_raw_data(np.array(data[i]), split_size = min_data_size, stride = 1)
			final_data = sub_videos if final_data is None else np.append(final_data, sub_videos, axis = 0)

		return final_data

	def load_data(self):
		counter, data = 0, []
		dir_size = len(os.listdir(self.folder))
		for filename in sorted(os.listdir(self.folder)):
			counter += 1
			if (self.file_list is None) or ((self.file_list is not None) and (filename in self.file_list)):
				print("[DataLoader - {0} / {1}] Trying to load video '{2}'.".format(counter, dir_size, filename))
				video_data = self.loader.load_video(filename = self.folder + '/' + filename)
				if video_data is not None:
					data += [video_data]

		return self.post_process(data)

class ReplayAttackLoader:
	def custom_shuffle(self, a, b):
		assert len(a) == len(b)
		p = np.random.permutation(len(a))
		return a[p], b[p]

	def load_data(self, folder, time):
		print("[ReplayAttackLoader] Loading ReplayAttack videos from '{0}/'.".format(folder))
		fake_fixed = DataLoader(folder = folder + "/attack/fixed", time = time).load_data()
		fake_hand = DataLoader(folder = folder + "/attack/hand", time = time).load_data()
		fake = np.append(fake_hand, fake_fixed, axis = 0)

		real = DataLoader(folder = folder + "/real", time = time).load_data()
		labels = np.append(np.zeros(fake.shape[0]), np.ones(real.shape[0]))
		return self.custom_shuffle(np.append(real, fake, axis = 0), to_categorical(labels, 2))

from keras.layers import Flatten, Dense, LSTM, Reshape, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

# Pred: 0 0 0 1
# True: 0 1 1 0
#  FP : 0 0 0 1
#  FN : 0 1 1 0

#Construir as métricas!
def build_cnn_model(frame_count, frame_rate):
	import keras.backend as K
	def FPR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_true, y_pred], axis = 0))

	def FNR(y_true, y_pred):
		y_true = K.argmax(y_true)
		y_pred = K.argmax(y_pred)
		return K.mean(K.all([1 - y_pred, y_true], axis = 0))

	model = Sequential()

	model.add(Reshape((frame_count // frame_rate, frame_rate, 3), input_shape = (frame_count, 3)))
	units = [16]
	for u in units:
		model.add(BatchNormalization())
		for i in range(2):
			model.add(Conv2D(filters = u, kernel_size = 3, strides = 1, padding = 'same'))
			model.add(Activation("elu"))

		model.add(MaxPooling2D(pool_size = 3, strides = 2))
		model.add(Activation("elu"))

	model.add(Flatten())
	# model.add(LSTM(units = 16))

	model.add(Dense(2, activation = "sigmoid"))

	model.compile(
		optimizer = Adam(lr = 1e-5),
		loss = "binary_crossentropy",
		metrics = ["accuracy", FNR, FPR]
	)

	model.summary()
	return model

import argparse

def build_parser():
	parser = argparse.ArgumentParser(description = "Código para treino das redes neurais utilizadas para detecção de ataques de apresentação.")
	parser.add_argument("dataset_folder", nargs = None, default = None, action = "store",
						help = "Localização da pasta do dataset. (Padrão: %(default)s)")
	parser.add_argument("process_folder", nargs = None, default = None, action = "store",
						help = "Localização da pasta onde serão (ou foram) armazenados os dados processados. (Padrão: %(default)s)")
	parser.add_argument("--time", dest = "time", nargs = "?", default = 5, action = "store", type = int,
						help = "Tempo (em segundos) de captura ou análise. (Padrão: %(default)s segundos)")
	parser.add_argument("--process", dest = "process_data", action = "store_true", default = False,
						help = "Define se serão obtidas novas médias a partir do dataset informado. (Padrão: %(default)s)")
	# parser.add_argument("--aproach", dest = "aproach", nargs = "?", default = ["DeHaan", "DeHaanPOS", "ICA"], action = "store",
	# 					help = "Abordagem no cálculo do rPPG. (Padrão: %(default)s)")
	return parser

from algorithms.de_haan_pos import DeHaanPOS

def add_algorithm_info(data, frame_rate):
	limit = 2000
	# data_after_processing = np.empty([data.shape[0], data.shape[1], data.shape[2] + 1])
	data_after_processing = np.empty([limit, data.shape[1], data.shape[2] + 1])
	for i in range(limit):
		if i > 0 and i % 500 == 0:
			print("[PostProcessing] {0} / {1} temporal series processed.".format(i, len(data)))

		rppg = DeHaanPOS(temporal_means = data[i]).measure_reference(frame_rate = frame_rate, window_size = int(frame_rate * 1.5))
		data_after_processing[i] = np.append(data[i], np.reshape(rppg, (rppg.shape[0], 1)), axis = 1)
	
	return data_after_processing
		
if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

	t_x, t_y, v_x, v_y = None, None, None, None
	frame_rate = 25 #Obter frame-rate a partir do dataset. Solução temporária.
	
	args = build_parser().parse_args()
	if args.process_data is True:
		loader = ReplayAttackLoader()
		t_x, t_y = loader.load_data(folder = "{0}/train".format(args.dataset_folder), time = args.time)
		v_x, v_y = loader.load_data(folder = "{0}/devel".format(args.dataset_folder), time = args.time)

		np.save("{0}/train_x.npy".format(args.process_folder), t_x)
		np.save("{0}/train_y.npy".format(args.process_folder), t_y)

		np.save("{0}/validation_x.npy".format(args.process_folder), v_x)
		np.save("{0}/validation_y.npy".format(args.process_folder), v_y)
	else:
		t_x = np.load("{0}/train_x.npy".format(args.process_folder))
		t_y = np.load("{0}/train_y.npy".format(args.process_folder))
		v_x = np.load("{0}/validation_x.npy".format(args.process_folder))
		v_y = np.load("{0}/validation_y.npy".format(args.process_folder))

	# print("Total de exemplos positivos nos dados de treino: {0}".format(np.sum(np.argmax(t_y, axis = 1))))
	# print("Tamanho dos dados de treino: {0}".format(len(t_y)))

	print("Train shape: {0}".format(t_x.shape))
	print("Validation shape: {0}".format(v_x.shape))

	model = build_cnn_model(frame_count = args.time * frame_rate, frame_rate = frame_rate)
	model.fit(x = t_x, y = t_y, validation_data = (v_x, v_y), batch_size = 8, epochs = 1)