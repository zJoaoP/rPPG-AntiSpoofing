from algorithms.default import DefaultStrategy
import matplotlib.pyplot as plt
import numpy as np

#https://pure.tue.nl/ws/portalfiles/portal/31563684/TBME_00467_2016_R1_preprint.pdf
#https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
class Wang(DefaultStrategy):
	@staticmethod
	def extract_rppg(temporal_means, frame_rate=25, window_size=None):
		def get_order(size):
			return (size - 6) // 3

		if window_size is None:
			window_size = int(frame_rate * 1.6)

		signal = np.zeros(len(temporal_means), dtype=np.float32)
		for t in range(len(temporal_means) - window_size):
			Cn = temporal_means[t:t+window_size].T
			if np.any(Cn.mean(axis=0) == 0.0):
				continue
			else:
				Cn = Cn / Cn.mean(axis=0)
			
			projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)
			S = np.matmul(projection_matrix, Cn)

			if np.std(S[1]) == 0.0:
				continue
			else:	
				h = S[0] + (np.std(S[0]) / np.std(S[1])) * S[1]
			
			signal[t:t+window_size] += h - h.mean()

		return DefaultStrategy.moving_average(signal, frame_rate // 4)
