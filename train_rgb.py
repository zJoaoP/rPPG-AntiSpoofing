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
		fake = DataLoader(folder = folder + "/attack/fixed", time = time).load_data() # Estou carregando apenas os vídeos com câmera fixa.
		real = DataLoader(folder = folder + "/real", time = time).load_data() # Estou carregando apenas os vídeos com câmera fixa.
		labels = np.append(np.zeros(fake.shape[0]), np.ones(real.shape[0]))
		return self.custom_shuffle(np.append(real, fake, axis = 0), to_categorical(labels, 2))

from keras.layers import Flatten, Dense, LSTM, Reshape, Conv1D, MaxPooling1D, BatchNormalization, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

def build_cnn_model(frame_count, frame_rate):
	model = Sequential()

	model.add(Reshape((frame_count // frame_rate, frame_rate), input_shape = (frame_count,)))
	units = [16]
	for u in units:
		model.add(BatchNormalization())
		model.add(Conv1D(filters = u, kernel_size = 11, padding = 'same'))
		model.add(MaxPooling1D(pool_size = 3, strides = 2))
		model.add(Activation("elu"))

	model.add(LSTM(units = 16))

	# model.add(Flatten())
	model.add(Dense(2, activation = "sigmoid"))

	model.compile(
		optimizer = Adam(lr = 1e-4),
		loss = "binary_crossentropy",
		metrics = ["accuracy"]
	)

	model.summary()
	return model
		
if __name__ == "__main__":
	time, frame_rate = [5, 25] # Generalizar o frame_rate?
	loader = ReplayAttackLoader()

	t_x, t_y = loader.load_data(folder = "./videos/ReplayAttack/train", time = 5)
	v_x, v_y = loader.load_data(folder = "./videos/ReplayAttack/devel", time = 5)

	# Verificar se é necessário processar ou apenas carregar. (argparse)
	np.save("./videos/train_x.npy", t_x)
	np.save("./videos/train_y.npy", t_y)

	np.save("./videos/validation_x.npy", v_x)
	np.save("./videos/validation_y.npy", v_y)

	print("Train shape: {0}".format(t_x.shape))
	print("Validation shape: {0}".format(v_x.shape))

	model = build_cnn_model(frame_count = time * frame_rate, frame_rate = frame_rate)
	model.fit(x = t_x[:, :, 1], y = t_y, validation_data = (v_x[:, :, 1], v_y), batch_size = 8, epochs = 500)