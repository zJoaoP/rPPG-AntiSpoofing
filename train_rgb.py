'''
	# Objetivo: Este arquivo possui (como objetivo principal) o treino de uma rede neural que recebe como entrada as features extraídas
	pelo algoritmo proposto por Gerard De Haan (CHROM) e exportará True (Em caso de ataque de apresentação) ou False (Caso contrário).

	# Etapas:
		1. Carregar os dados. (OK)
			1.1 - Definir classe para carregamento de dados. (Tentar desenvolver de modo que seja possível escolher quais arquivos serão carregados)
			1.2 - Escrever código simples para 'argparse'.
			1.3 - De que modo a estratégia será passada para a rede? Qual a saída? Será sempre um vetor com o tamanho da quantidade de frames?
		
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
		self.face_predictor = LandmarkPredictor()
		self.roi_extractor = roi_extractor
		self.time = time

	def is_valid_video(self, filename):
		valid_extensions = ["avi", "mp4", "mov"]
		extension = filename.split('.')[-1]
		return extension in valid_extensions

	# def split_raw_data(self, raw_data, split_size = 15 * 30, stride = 1):
	# 	while index + self.time < len(raw_data):
	# 		data = np.array([raw_data[index : index + self.time]]) if (data is None) else np.append(data, [raw_data[index : index + self.time]], axis = 0)
	# 		index += stride

	# 	return data

	def load_video(self, filename):
		if self.is_valid_video(filename):
			stream = cv2.VideoCapture(filename)
			frame_rate, current_frame = int(stream.get(cv2.CAP_PROP_FPS)), 0
			means = None
			while current_frame < frame_rate * self.time:
				success, frame = stream.read()
				if not success:
					break

				face_rect = self.face_predictor.detect_face(image = frame)
				if face_rect is not None:
					face_left, face_right, face_top, face_bottom = face_rect.left(), face_rect.right(), face_rect.top(), face_rect.bottom()
					cut_face = frame[face_top : face_bottom, face_left : face_right, :]
					landmarks = self.face_predictor.detect_landmarks(image = cut_face)

					roi = self.roi_extractor.extract_roi(cut_face, landmarks)
					
					if current_frame % 60 == 0 and current_frame > 0:
						print("[VideoLoader] {0} frames extracted from '{1}'.".format(current_frame, filename))
					
					roi = roi.astype(np.float32)
					roi[roi < 5.0] = np.nan

					r = np.nanmean(roi[:, :, 2])
					g = np.nanmean(roi[:, :, 1])
					b = np.nanmean(roi[:, :, 0])

					means = np.array([[r, g, b]]) if (means is None) else np.append(means, [[r, g, b]], axis = 0)

					current_frame += 1

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

	def load_data(self):
		data_size = len(self.file_list) if self.file_list is not None else len(os.listdir(self.folder))
		data, k = [None, 0]
		for filename in sorted(os.listdir(self.folder)):
			if (self.file_list is None) or ((self.file_list is not None) and (filename in self.file_list)):
				video_data = self.loader.load_video(self.folder + '/' + filename)

				if (video_data is not None) and (data is None):
					data = np.empty([data_size, len(video_data), self.num_channels])				

				if video_data is not None:
					video_data = video_data if video_data.shape[0] == data.shape[1] else video_data[:data.shape[1], :]
					data[k] = video_data
					k += 1

		print(data.shape)
		return data
		
if __name__ == "__main__":
	fake_data = DataLoader(folder = "./videos/Fake", time = 10).load_data()
	real_data = DataLoader(folder = "./videos/Real", time = 10).load_data()

	print(fake_data.shape)
	print(real_data.shape)