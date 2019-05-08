from utils.extractor import CheeksOnly, FaceWithoutEyes, CheeksAndNose

from algorithms.default import DefaultStrategy
import matplotlib.pyplot as plt
import numpy as np
import cv2

class DeHaan(DefaultStrategy):
	def __init__(self):
		self.temporal_means = None

	def process(self, frame, landmarks):
		frame = CheeksAndNose().extract_roi(frame, landmarks)
		frame = np.array(frame, dtype = np.float32)

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

		signal = np.zeros(len(self.temporal_means), dtype = np.float32)
		window_stride = window_size // 2
		for j in range(0, len(self.temporal_means) - window_size, window_stride):
			if j + window_size > len(self.temporal_means):
				break

			r = self.temporal_means[j : j + window_size, 0] / np.mean(self.temporal_means[j : j + window_size, 0])
			g = self.temporal_means[j : j + window_size, 1] / np.mean(self.temporal_means[j : j + window_size, 1])
			b = self.temporal_means[j : j + window_size, 2] / np.mean(self.temporal_means[j : j + window_size, 2])

			Xs = (3.0 * r) - (2.0 * g)
			Ys = (1.5 * r) + (1.0 * g) - (1.5 * b)

			Xf = self.bandpass_filter(Xs, frame_rate = frame_rate, min_freq = 0.6, max_freq = 4.0, order = get_order(window_size))
			Yf = self.bandpass_filter(Ys, frame_rate = frame_rate, min_freq = 0.6, max_freq = 4.0, order = get_order(window_size))

			alpha = np.std(Xf) / np.std(Yf)

			window_signal = Xf - (alpha * Yf)
			signal[j : j + window_size] += (window_signal - np.mean(window_signal)) / np.std(window_signal)

		return signal

	def show_results(self, frame_rate = 30, window_size = 60, plot = True):
		signal = self.measure_reference(frame_rate = frame_rate, window_size = window_size)
		x, y = self.get_fft(signal, frame_rate = frame_rate)

		if plot == True:

			plt.subplot(3, 1, 1)
			
			plt.title("DeHaan")
			plt.plot(self.temporal_means[:, 0], 'r')
			plt.plot(self.temporal_means[:, 1], 'g')
			plt.plot(self.temporal_means[:, 2], 'b')

			plt.subplot(3, 1, 2)
			plt.plot(signal)

			plt.subplot(3, 1, 3)
			plt.plot(x, y)
	
			plt.show()

		print("[DeHaan] {0:2f} bpm.".format(60.0 * x[np.argmax(y)]))
