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

	def measure_reference(self, frame_rate = 30, window_size = 45):
		window_count = len(self.temporal_means) // window_size
		windows = np.empty([window_count, window_size, 3])

		for j, k in zip(range(0, len(self.temporal_means), window_size), range(window_count)):
			if j + window_size > len(self.temporal_means):
				break

			windows[k] = self.temporal_means[j : j + window_size]
			k += 1

		signal = np.zeros([len(self.temporal_means) - (len(self.temporal_means) % window_size)], dtype = np.float32) #Sobreposição de janelas.
		for j, k in zip(range(0, len(self.temporal_means), window_size), range(window_count)):
			if j + window_size > len(self.temporal_means):
				break

			b, g, r = [(windows[k][:, i] - np.mean(windows[k][:, i])) / np.std(windows[k][:, i]) for i in range(3)]

			Xs = 3.0 * r + 2.0 * g
			Ys = 1.5 * r + 1.0 * g - 1.5 * b

			Xf = self.bandpass_filter(Xs, frame_rate = frame_rate, min_freq = 0.7, max_freq = 4.0, order = 17)
			Yf = self.bandpass_filter(Ys, frame_rate = frame_rate, min_freq = 0.7, max_freq = 4.0, order = 17)

			alpha = np.std(Xf) / np.std(Yf)
			window_signal = Xf - (alpha * Yf) #Normalizar com mean e std???

			window_signal = (window_signal - np.mean(window_signal)) / np.std(window_signal) #Padronização.

			signal[j : j + window_size] = window_signal
			k += 1

		return signal

	def show_results(self, frame_rate = 30, window_size = 60):
		reference = self.measure_reference(frame_rate = frame_rate, window_size = window_size)
		
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