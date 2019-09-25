from utils.extractor import CheeksOnly, FaceWithoutEyes, CheeksAndNose

from algorithms.default import DefaultStrategy
import matplotlib.pyplot as plt
import numpy as np
import cv2

class DeHaan(DefaultStrategy):
	@staticmethod
	def extract_rppg(temporal_means, frame_rate=30):
		def get_order(size):
			return (size - 6) // 3

		signal = np.zeros(len(temporal_means), dtype=np.float32)

		window_size = int(frame_rate * 1.6)

		for i in range(3):
			temporal_means[:, i] = (temporal_means[:, i] / (np.mean(temporal_means[:, i]) + np.finfo(np.float32).eps)) - 1.0

			temporal_means[:, i] = DefaultStrategy.detrend(temporal_means[:, i])
			temporal_means[:, i] = DefaultStrategy.bandpass_filter(temporal_means[:, i],
																   frame_rate=frame_rate,
																   min_freq=0.6,
																   max_freq=4.0,
																   order=get_order(window_size))

			# self.temporal_means[:, i] = self.detrend(self.temporal_means[:, i])

		for j in range(0, len(temporal_means), window_size // 2):
			if j + window_size > len(temporal_means):
				break

			r, g, b = [temporal_means[j : j + window_size, i] for i in range(3)]

			Xs = (3.0 * r) - (2.0 * g)
			Ys = (1.5 * r) + (1.0 * g) - (1.5 * b)

			Xf = DefaultStrategy.bandpass_filter(Xs, frame_rate=frame_rate, min_freq=0.6, max_freq=4.0, order=get_order(window_size))
			Yf = DefaultStrategy.bandpass_filter(Ys, frame_rate=frame_rate, min_freq=0.6, max_freq=4.0, order=get_order(window_size))

			alpha = np.std(Xf) / np.std(Yf)

			signal[j:j + window_size] += np.hanning(window_size) * (Xf - (alpha * Yf))

		return (signal - np.mean(signal)) / np.std(signal)
		# return signal
