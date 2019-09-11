from utils.extractor import CheeksOnly, FaceWithoutEyes, CheeksAndNose
from algorithms.default import DefaultStrategy

import matplotlib.pyplot as plt
import numpy as np

#https://pure.tue.nl/ws/portalfiles/portal/31563684/TBME_00467_2016_R1_preprint.pdf
#https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
class Wang(DefaultStrategy):
	@staticmethod
	def extract_rppg(temporal_means, frame_rate=30, window_size=None):
		def get_order(size):
			return (size - 6) // 3

		if window_size is None:
			window_size = int(frame_rate * 1.6)

		signal = np.zeros(len(temporal_means), dtype=np.float32)
		num_frames = len(temporal_means)

		# for i in range(3):
		# 	temporal_means[:, i] -= np.mean(temporal_means[:, i])
		# 	temporal_means[:, i] = DefaultStrategy.detrend(temporal_means[:, i])

		for n in range(len(temporal_means)):
			Cn = temporal_means
			m = n - window_size + 1
			if m > 0:
				Cin = Cn[m : n, :]
				S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cin.T)
				std_1 = np.std(S[1])
				if std_1 > 0.0:
					H = S[0] + (np.std(S[0]) / np.std(S[1])) * S[1]
				else:
					H = S[0]

				signal[m : n] += (H - np.mean(H))

		signal = DefaultStrategy.bandpass_filter(signal,
												frame_rate=frame_rate,
												min_freq=0.7,
												max_freq=4.0, 
												order=get_order(len(signal)))
		
		return (signal - signal.mean()) / signal.std()