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

	# Cortar dados de forma que vídeos maiores que o necessário se transformem em mais que apenas uma sequência. (E sim em várias)
	# Um vídeo de 450 frames (a 30 fps) pode ser cortado em 150 * 300 frames.
	def split_raw_data(self, raw_data, split_size, stride = 1):
		data, i = [], 0
		while i + split_size < len(raw_data):
			data += [raw_data[i : i + split_size, :]]
			i += stride

		return np.array(data)

	def load_video(self, filename, max_cut_size):
		if self.is_valid_video(filename):
			stream = cv2.VideoCapture(filename)
			face_predictor = LandmarkPredictor()
			frame_rate, current_frame = int(stream.get(cv2.CAP_PROP_FPS)), 0
			means = []

			cut_size = min(self.time * frame_rate, max_cut_size)
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

	def get_min_frame_count(self):
		min_frame_count = (1 << 32)
		for filename in sorted(os.listdir(self.folder)):
			_, frame_count = self.loader.get_video_info(filename)
			min_frame_count = min_frame_count if frame_count <= 0 else min(min_frame_count, frame_count)

		return min_frame_count

	def load_data(self):
		min_frame_count = self.get_min_frame_count()
		data = []
		for filename in sorted(os.listdir(self.folder)):
			if (self.file_list is None) or ((self.file_list is not None) and (filename in self.file_list)):
				video_data = self.loader.load_video(filename = self.folder + '/' + filename, max_cut_size = min_frame_count)
				if video_data is not None:
					data += [video_data]

		# print(np.array(video_data).shape)
		# print(data.shape)
		print(len(data))
		return data
		
if __name__ == "__main__":
	fake_data = DataLoader(folder = "./videos/Fake", time = 5).load_data()
	real_data = DataLoader(folder = "./videos/Real", time = 5).load_data()

	# print(fake_data.shape)
	# print(real_data.shape)