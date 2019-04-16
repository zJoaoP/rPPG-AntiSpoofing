from algorithms.default import DefaultStrategy

import matplotlib.pyplot as plt
import numpy as np
import cv2

class DeHaan(DefaultStrategy):
	def __init__(self):
		self.temporal_means = None

	def process(self, frame):
		frame = np.array(frame, dtype = np.float32)
		frame[frame == 0.0] = np.nan
		means = np.nanmean(frame, axis = (0, 1))

		if self.temporal_means is None:
			self.temporal_means = np.array([means])
		else:
			self.temporal_means = np.append(self.temporal_means, [means], axis = 0)

	def measure_reference(self, window_size = 30):
		r_norm = self.moving_window_normalization(self.temporal_means[:, 2], window_size)
		g_norm = self.moving_window_normalization(self.temporal_means[:, 1], window_size)
		b_norm = self.moving_window_normalization(self.temporal_means[:, 0], window_size)

		Xs = 3.0 * r_norm + 2.0 * g_norm
		Ys = 1.5 * r_norm + 1.0 * g_norm - 1.5 * b_norm

		Xf = self.bandpass_filter(Xs, low = 0.4, high = 3.0)
		Yf = self.bandpass_filter(Ys, low = 0.4, high = 3.0)

		alpha = np.std(Xf) / np.std(Yf)

		return Xf - (alpha * Yf)

	def show_results(self, window_size = 30):
		reference = self.measure_reference(window_size = window_size)
		plt.subplot(2, 1, 1)
		plt.plot(reference)

		reference_fourier = np.fft.rfft(reference)
		plt.subplot(2, 1, 2)
		plt.plot(np.abs(reference_fourier))

		plt.show()