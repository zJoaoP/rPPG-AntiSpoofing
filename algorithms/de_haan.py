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
		r = self.moving_window_normalization(self.temporal_means[:, 2], window_size)
		g = self.moving_window_normalization(self.temporal_means[:, 1], window_size)
		b = self.moving_window_normalization(self.temporal_means[:, 0], window_size)

		# r = self.temporal_means[:, 2]
		# g = self.temporal_means[:, 1]
		# b = self.temporal_means[:, 0]

		Xs = 3.0 * r + 2.0 * g
		Ys = 1.5 * r + 1.0 * g - 1.5 * b

		Xf = self.bandpass_filter(Xs)
		Yf = self.bandpass_filter(Ys)

		alpha = np.std(Xf) / np.std(Yf)
		return Xf - (alpha * Yf)

	def show_results(self, frame_rate = 30, window_size = 30):
		# https://www.arduino.cc/reference/en/language/functions/math/map/
		def map_range(x, in_min, in_max, out_min, out_max):
			return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;

		reference = self.measure_reference(window_size = window_size)
		
		plt.subplot(3, 1, 1)
		plt.plot(self.temporal_means[:, 2], 'r')
		plt.plot(self.temporal_means[:, 1], 'g')
		plt.plot(self.temporal_means[:, 0], 'b')

		plt.subplot(3, 1, 2)
		plt.plot(reference)

		x, y = self.get_fft(reference, frame_rate = frame_rate)

		print("[DeHaan] {0:2f} beats per minute.".format(60.0 * x[np.argmax(y)]))
		plt.subplot(3, 1, 3)
		plt.plot(x, y)

		plt.show()