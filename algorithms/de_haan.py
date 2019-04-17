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

		Xf = self.bandpass_filter(Xs)
		Yf = self.bandpass_filter(Ys)

		alpha = np.std(Xf) / np.std(Yf)

		return Xf - (alpha * Yf)

	def show_results(self, window_size = 45):
		reference = self.measure_reference(window_size = window_size)
		plt.subplot(3, 1, 1)
		plt.plot(self.temporal_means[:, 2], 'r')
		plt.plot(self.temporal_means[:, 1], 'g')
		plt.plot(self.temporal_means[:, 0], 'b')

		plt.subplot(3, 1, 2)
		plt.plot(reference)

		reference_fourier = np.abs(np.fft.rfft(reference, norm = "ortho"))
		contribution = 1.0 / len(reference_fourier)
		print(np.argmax(reference_fourier))
		print(reference_fourier[np.argmax(reference_fourier)])
		print(len(reference_fourier))
		highest_frequency = 0.0

		plt.subplot(3, 1, 3)
		plt.title("FFT of Signal({0:2f} bpm)".format(highest_frequency * 60.0))
		plt.plot(reference_fourier)


		plt.show()