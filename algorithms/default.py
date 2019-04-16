from scipy import signal
import numpy as np

class DefaultStrategy:	
	def moving_window_normalization(self, values, window_size):
		windowed_normalization = np.zeros([len(values)])
		left, right = (0, window_size)
		for i in range(len(values)):
			windowed_normalization[i] = values[i] / np.mean(values[i - left : i + right])
			if (left + 1 < right) or (i + right == len(values)):
				left += 1
				right -= 1

		return windowed_normalization

	def filter_signal(self, y, threshold, mode = "low"):
		w = threshold / (len(y) / 2.0)
		b, a = signal.butter(3, w, mode)
		return signal.filtfilt(b, a, y)

	def bandpass_filter(self, y, low, high):
		return self.filter_signal(self.filter_signal(y, high, 'low'), low, 'high')
