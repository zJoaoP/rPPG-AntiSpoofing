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

	# https://gitlab.idiap.ch/bob/bob.rppg.base/blob/master/bob/rppg/base/utils.py
	# def build_bandpass_filter(self, frame_rate, order, min_freq = 0.7, max_freq = 4.0):
	# 	from scipy.signal import firwin
		
	# 	nyq = frame_rate / 2.0
	# 	num_taps = order + 1
	# 	return firwin(num_taps, [min_freq / nyq, max_freq / nyq], pass_zero = False)

	# def bandpass_filter(self, y, frame_rate = 30, min_freq = 0.6, max_freq = 3.0):
	# 	from scipy.signal import filtfilt
		
	# 	band_pass_filter = self.build_bandpass_filter(frame_rate, 128, min_freq, max_freq)
	# 	# print(band_pass_filter)
	# 	return filtfilt(band_pass_filter, np.array([1]), y)

	def filter_signal(self, y, threshold, mode = "low"):
		w = threshold / (len(y) / 2.0)
		b, a = signal.butter(4, w, mode)
		return signal.filtfilt(b, a, y)

	def bandpass_filter(self, y, low = 0.5, high = 4.0):
		return self.filter_signal(self.filter_signal(y, high, 'low'), low, 'high')