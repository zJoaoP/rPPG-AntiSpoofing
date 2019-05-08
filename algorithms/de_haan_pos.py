from utils.extractor import CheeksOnly, FaceWithoutEyes, CheeksAndNose
from algorithms.default import DefaultStrategy

import matplotlib.pyplot as plt
import numpy as np

#https://pure.tue.nl/ws/portalfiles/portal/31563684/TBME_00467_2016_R1_preprint.pdf
#https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
class DeHaanPOS(DefaultStrategy):
	def __init__(self):
		self.temporal_means = None

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
		signal = np.zeros(len(self.temporal_means), dtype = np.float64)
		num_frames = len(self.temporal_means)

		window_stride = 1
		for f_id in range(0, num_frames, window_stride):
				if f_id + window_size >= num_frames:
					break

				C = self.temporal_means[f_id : f_id + window_size, :].T
				
				# https://www.youtube.com/watch?v=GMN1A8Wfwto
				C_mean = np.mean(C, axis = 1)
				C_mean_diag = np.diag(C_mean)
				C_mean_diag_inv = np.linalg.inv(C_mean_diag)
				
				Cn = np.matmul(C_mean_diag_inv, C)

				S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
				P = S[0] + (np.std(S[0]) / np.std(S[1])) * S[1]

				signal[f_id : f_id + window_size] += P - np.mean(P)

		return (signal - np.mean(signal)) / np.std(signal)

	def show_results(self, frame_rate = 30, window_size = 60, plot = True):
		signal = self.measure_reference(frame_rate = frame_rate, window_size = window_size)
		filtered_signal = self.bandpass_filter(signal, frame_rate = frame_rate, min_freq = 0.6, max_freq = 4.0, order = 128)

		x, y = self.get_fft(signal, frame_rate = frame_rate)
		x_f, y_f = self.get_fft(filtered_signal, frame_rate = frame_rate)

		if plot == True:
			plt.title("POS")
			
			plt.subplot(3, 1, 1)
			plt.plot(self.temporal_means[:, 0], 'r')
			plt.plot(self.temporal_means[:, 1], 'g')
			plt.plot(self.temporal_means[:, 2], 'b')

			plt.subplot(3, 1, 2)
			plt.plot(signal)
			plt.plot(filtered_signal)

			plt.subplot(3, 1, 3)
			plt.plot(x, y)
			plt.plot(x_f, y_f)

			plt.show()

		print("[DeHaanPOS - Without BPF] {0:2f} bpm.".format(60.0 * x[np.argmax(y)]))
		print("[DeHaanPOS - With BPF] {0:2f} bpm.".format(60.0 * x_f[np.argmax(y_f)]))

		plt.show()