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
		def get_order(size):
			return (size - 6) // 3

		signal = np.zeros(len(self.temporal_means), dtype = np.float32)

		r = self.temporal_means[:, 2] / np.mean(self.temporal_means[:, 2])
		g = self.temporal_means[:, 1] / np.mean(self.temporal_means[:, 1])
		b = self.temporal_means[:, 0] / np.mean(self.temporal_means[:, 0])

		Xs = (3.0 * r) + (2.0 * g)
		Ys = (1.5 * r) + (1.0 * g) - (1.5 * b)

		print("ORDER: {0}".format(get_order(len(self.temporal_means))))

		Xf = self.bandpass_filter(Xs, frame_rate = frame_rate, min_freq = 0.7, max_freq = 4.0, order = get_order(len(self.temporal_means)))
		Yf = self.bandpass_filter(Ys, frame_rate = frame_rate, min_freq = 0.7, max_freq = 4.0, order = get_order(len(self.temporal_means)))
		
		window_stride = window_size // 2
		for j in range(0, len(self.temporal_means) - window_size, window_stride):
			xf_window = Xf[j : j + window_size]
			yf_window = Yf[j : j + window_size]

			alpha = np.std(xf_window) / np.std(yf_window)

			window_signal = xf_window - (alpha * yf_window)
			window_signal *= np.hanning(window_size)

			signal[j : j + window_size] += (window_signal - np.mean(window_signal)) / np.std(window_signal)

		return signal

	def show_results(self, frame_rate = 30, window_size = 60, plot = True):
		signal = self.measure_reference(frame_rate = frame_rate, window_size = window_size)
		x, y = self.get_fft(signal, frame_rate = frame_rate)

		if plot == True:

			plt.subplot(3, 1, 1)
			
			plt.title("DeHaan")
			plt.plot(self.temporal_means[:, 2], 'r')
			plt.plot(self.temporal_means[:, 1], 'g')
			plt.plot(self.temporal_means[:, 0], 'b')

			plt.subplot(3, 1, 2)
			plt.plot(signal)

			plt.subplot(3, 1, 3)
			plt.plot(x, y)
	
			plt.show()

		print("[DeHaan] {0:2f} beats per minute.".format(60.0 * x[np.argmax(y)]))
