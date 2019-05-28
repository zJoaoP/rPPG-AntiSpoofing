from algorithms.default import DefaultStrategy
from utils.extractor import CheeksOnly, FaceWithoutEyes, Nose

import matplotlib.pyplot as plt
import numpy as np

class Green(DefaultStrategy):
	def __init__(self, green_means = None):
		self.green_means = green_means

	def process(self, frame, landmarks):
		frame = FaceWithoutEyes().extract_roi(frame, landmarks)
		frame = np.array(frame, dtype = np.float32)

		frame[frame < 5.0] = np.nan
		green_mean = np.array([np.nanmean(frame[:, :, 1])])

		self.green_means = green_mean if self.green_means is None else np.append(self.green_means, green_mean)

	def measure_reference(self, frame_rate = 30, window_size = 45):
		def get_order(size):
			return (size - 6) // 3

		signal = self.green_means - np.mean(self.green_means)
		signal = self.detrend(signal)
		signal = self.moving_average(signal, 30)
		signal = self.bandpass_filter(signal, frame_rate = frame_rate, min_freq = 0.6, max_freq = 4.0, order = get_order(len(signal)))
		return signal

	def show_results(self, frame_rate = 30, window_size = 60, plot = True):
		signal = self.measure_reference()
		x, y = self.get_fft(signal, frame_rate = frame_rate)

		plt.subplot(2, 1, 1)
		plt.plot(signal, 'g')

		plt.subplot(2, 1, 2)
		plt.plot(x, y)

		plt.show()