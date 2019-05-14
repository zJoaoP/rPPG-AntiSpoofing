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
	def __init__(self, window_size = 15, roi_extractor = CheeksAndNose()):
		self.face_predictor = LandmarkPredictor()
		self.roi_extractor = roi_extractor
		self.window_size = window_size

	def is_valid_video(self, filename):
		valid_extensions = ["avi", "mp4", "mov"]
		extension = filename.split('.')[-1]
		return extension in valid_extensions

	def split_raw_data(self, raw_data, stride = 1):
		index, data = [0, None]
		while index + self.window_size < len(raw_data):
			data = np.array([raw_data[index : index + self.window_size]]) if (data is None) else np.append(data, [raw_data[index : index + self.window_size]], axis = 0)
			index += stride

		return data

	def load_video(self, filename):
		if self.is_valid_video(filename):
			stream = cv2.VideoCapture(filename)
			current_frame = 0
			means = None
			while True:
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

			return self.split_raw_data(means, stride = 15)
		else:
			print("[VideoLoader] Skipping '{0}'. Reason: Invalid Video! (It's a folder??)".format(filename))
			return None

class DataLoader:
	def __init__(self, folder, file_list = None, window_size = 15, roi_extractor = CheeksAndNose()):
		self.roi_extractor = roi_extractor
		self.window_size = window_size
		self.file_list = file_list
		self.folder = folder

		self.loader = VideoLoader(window_size = window_size, roi_extractor = roi_extractor)

	def load_data(self):
		data = None
		for filename in sorted(os.listdir(self.folder)):
			if (self.file_list is None) or ((self.file_list is not None) and (filename in self.file_list)):
				video_data = self.loader.load_video(self.folder + '/' + filename)
				if video_data is not None:					
					data = video_data if (data is None) else np.append(data, video_data, axis = 0)

		print(data.shape)
		return data
		
if __name__ == "__main__":
	DataLoader(folder = "./videos").load_data()