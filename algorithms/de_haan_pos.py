from utils.extractor import CheeksOnly, FaceWithoutEyes, CheeksAndNose
from algorithms.default import DefaultStrategy

import matplotlib.pyplot as plt
import numpy as np

#https://pure.tue.nl/ws/portalfiles/portal/31563684/TBME_00467_2016_R1_preprint.pdf
#https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
class DeHaanPOS(DefaultStrategy):
	def __init__(self, temporal_means = None):
		self.temporal_means = temporal_means

	def process(self, frame, landmarks):
		frame = CheeksAndNose().extract_roi(frame, landmarks)
		frame = np.array(frame, dtype = np.float64)
		frame[frame == 0.0] = np.nan

		r = np.nanmean(frame[:, :, 2])
		g = np.nanmean(frame[:, :, 1])
		b = np.nanmean(frame[:, :, 0])

		if self.temporal_means is None:
			self.temporal_means = np.array([[r, g, b]])
		else:
			self.temporal_means = np.append(self.temporal_means, [[r, g, b]], axis = 0)

	def measure_reference(self, frame_rate = 30, window_size = 45):
		def get_order(size):
			return (size - 6) // 3

		signal = np.zeros(len(self.temporal_means), dtype = np.float64)
		num_frames = len(self.temporal_means)

		def detrend_signal(signal):
			signal = signal - np.mean(signal)
			return self.detrend(signal)

		for i in range(3):
			self.temporal_means[:, i] -= np.mean(self.temporal_means[:, i])
			self.temporal_means[:, i] = self.detrend(self.temporal_means[:, i])

		for n in range(len(self.temporal_means)):
			Cn = self.temporal_means
			m = n - window_size + 1
			if m > 0:
				Cin = Cn[m : n, :]
				S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cin.T)
				H = S[0] + (np.std(S[0]) / np.std(S[1])) * S[1]
				signal[m : n] += (H - np.mean(H))

		return self.bandpass_filter(signal, frame_rate = frame_rate, min_freq = 0.7, max_freq = 4.0, order = get_order(window_size))

	def show_results(self, frame_rate = 30, window_size = 60, plot = True):
		signal = self.measure_reference(frame_rate = frame_rate, window_size = window_size)
		x, y = self.get_fft(signal, frame_rate = frame_rate)

		if plot == True:
			plt.title("POS")
			
			plt.subplot(2, 1, 1)
			plt.plot(self.temporal_means[:, 0], 'r')
			plt.plot(self.temporal_means[:, 1], 'g')
			plt.plot(self.temporal_means[:, 2], 'b')

			plt.subplot(2, 1, 2)
			plt.plot(x, y)

			plt.show()

		print("[DeHaanPOS - Without BPF] {0:2f} bpm.".format(60.0 * x[np.argmax(y)]))
		print("[DeHaanPOS - With BPF] {0:2f} bpm.".format(60.0 * x_f[np.argmax(y_f)]))

		plt.show()