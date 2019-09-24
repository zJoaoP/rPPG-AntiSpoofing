from algorithms.default import DefaultStrategy
import matplotlib.pyplot as plt
import numpy as np

class PBV(DefaultStrategy):
	@staticmethod
	def extract_rppg(temporal_means, frame_rate=25, window_size=None):
		def get_order(size):
			return (size - 6) // 3

		if window_size is None:
			window_size = int(frame_rate * 1.6)

		channels = temporal_means.T
		for i, c in enumerate(channels):
			c_overlap = np.zeros(len(c))
			for j in range(0, len(c), window_size // 2):
				if j + window_size > len(c):
					break

				c_window = c[j:j+window_size]
				# if c_window.mean() != 0:
				c_window = (c_window / c_window.mean()) - 1.0
				c_window = DefaultStrategy.detrend(c_window)

				c_window = DefaultStrategy.bandpass_filter(c_window,
															frame_rate=frame_rate,
															min_freq=0.6,
															max_freq=4.0,
															order=get_order(len(c_window)))

				c_overlap[j:j+window_size] += c_window

			channels[i] = c_overlap

		signal = np.zeros(len(channels[0]))
		for i in range(len(signal)):
			signal[i] = max(channels[j][i] for j in range(3)) - min(channels[j][i] for j in range(3))
			# signal[i] = max(channels[j][i] for j in range(3))	

		signal = DefaultStrategy.bandpass_filter(signal,
												frame_rate=frame_rate,
												min_freq=0.6,
												max_freq=4.0,
												order=get_order(len(signal)))

		return signal
