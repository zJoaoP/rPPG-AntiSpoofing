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
			window_size = frame_rate

		signal = np.zeros(len(temporal_means))
		for n in range(len(temporal_means)):
			Cn = temporal_means
			m = n - window_size + 1
			if m > 0:
				Cin = Cn[m : n, :]
				S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cin.T)
				if np.std(S[1]) == 0.0:
					continue
				else:
					H = S[0] + (np.std(S[0]) / np.std(S[1])) * S[1]
				
				signal[m : n] += (H - np.mean(H))

		signal = DefaultStrategy.bandpass_filter(signal,
												frame_rate=frame_rate,
												min_freq=0.6,
												max_freq=4.0,
												order=get_order(len(signal)))
		
		return (signal - signal.mean()) / signal.std()
