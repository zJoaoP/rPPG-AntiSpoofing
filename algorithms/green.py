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
		signal = self.green_means - np.mean(self.green_means)
		return signal

	def show_results(self, frame_rate = 30, window_size = 60, plot = True):
		signal = self.measure_reference()

		detrended_signal = self.detrend(signal)
		average_signal = self.moving_average(detrended_signal, 15)
		bandpass_signal = self.bandpass_filter(average_signal, frame_rate = frame_rate, min_freq = 0.6, max_freq = 4.0, order = 64)
		x, y = self.get_fft(bandpass_signal, frame_rate = frame_rate)

		plt.subplot(5, 1, 1)
		plt.plot(self.green_means, 'g')

		plt.subplot(5, 1, 2)
		plt.plot(detrended_signal, 'g')

		plt.subplot(5, 1, 3)
		plt.plot(average_signal, 'y')

		plt.subplot(5, 1, 4)
		plt.plot(bandpass_signal)

		plt.subplot(5, 1, 5)
		plt.plot(x, y)

		plt.show()