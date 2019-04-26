from algorithms.default import DefaultStrategy

import matplotlib.pyplot as plt
import numpy as np

#https://pure.tue.nl/ws/portalfiles/portal/31563684/TBME_00467_2016_R1_preprint.pdf
#https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
class DeHaanPOS(DefaultStrategy):
	def __init__(self):
		self.temporal_means = None

	def process(self, frame):
		frame = np.array(frame, dtype = np.float64)
		frame[frame == 0.0] = np.nan

		b = np.nanmean(frame[:, :, 0])
		g = np.nanmean(frame[:, :, 1])
		r = np.nanmean(frame[:, :, 2])

		if self.temporal_means is None:
			self.temporal_means = np.array([[b, g, r]])
		else:
			self.temporal_means = np.append(self.temporal_means, [[b, g, r]], axis = 0)

	def measure_reference(self, frame_rate = 30, window_size = 45):
		signal = np.zeros(len(self.temporal_means), dtype = np.float64)
		num_frames = len(self.temporal_means)

		C = self.temporal_means.T
		for f_id in range(num_frames):
			if f_id + window_size < num_frames:
				# Possível falha na precisão ao realizar operações sem produto de matrizes.

				C = self.temporal_means[f_id : f_id + window_size, :].T
				C_mean = np.mean(C, axis = 1, dtype = np.float64)
				C_mean_diag = np.diag(C_mean)
				C_mean_diag_inv = np.linalg.inv(C_mean_diag)
				Cn = np.matmul(C_mean_diag_inv, C)

				S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
				H = S[0] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1]

				signal[f_id : f_id + window_size] += (H - np.mean(H)) / np.std(H)

		return signal

	def show_results(self, frame_rate = 30, window_size = 60, plot = True):
		signal = self.measure_reference(frame_rate = frame_rate, window_size = window_size)
		x, y = self.get_fft(signal, frame_rate = frame_rate)

		if plot == True:
			plt.subplot(2, 1, 1)
			plt.plot(signal)

			plt.subplot(2, 1, 2)
			plt.plot(x, y)

			plt.show()

		print("[DeHaanPOS] {0:2f} beats per minute.".format(60.0 * x[np.argmax(y)]))

		plt.show()