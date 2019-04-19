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

	def get_fft(self, y, frame_rate = 30):
		sample_rate = 1.0 / float(frame_rate)
		sample_count = len(y)

		yf = np.fft.fft(y)
		xf = np.linspace(0.0, 1.0 / (2.0 * sample_rate), sample_count // 2)
		return xf, 2.0 / sample_count * np.abs(yf[0 : sample_count // 2])

	# https://gitlab.idiap.ch/bob/bob.rppg.base/blob/master/bob/rppg/base/utils.py
	def build_bandpass_filter(self, fs, order, min_freq = 0.7, max_freq = 4.0):
		from scipy.signal import firwin 
		
		nyq = fs / 2.0
		numtaps = order + 1
		return firwin(numtaps, [min_freq/nyq, max_freq/nyq], pass_zero=False)

	def bandpass_filter(self, data, frame_rate = 30, min_freq = 0.7, max_freq = 4.0, order = 3):
		from scipy.signal import filtfilt
		return filtfilt(self.build_bandpass_filter(fs = frame_rate, order = order, min_freq = min_freq, max_freq = max_freq), np.array([1]), data)