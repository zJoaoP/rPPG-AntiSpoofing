from scipy import signal
import numpy as np

class DefaultStrategy:
	@staticmethod
	def extract_means(frame):
		nan_frame = frame
		nan_frame[nan_frame == 0.0] = np.nan

		r = np.nanmean(nan_frame[:, :, 2])
		g = np.nanmean(nan_frame[:, :, 1])
		b = np.nanmean(nan_frame[:, :, 0])

		return [r, g, b]

	# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=979357
	@staticmethod
	def detrend(y):
		from scipy.sparse import identity
		from scipy.sparse import spdiags

		T = len(y)
		l = 10
		I = identity(T)
		
		data = np.ones(shape = [T - 2, 1]) * np.array([1, -2, 1])
		D2 = spdiags(data.T, np.array([0, 1, 2]), T - 2, T)

		operations = I + (l ** 2) * (D2.T * D2)
		inversion = np.linalg.inv(operations.toarray())

		y_stat = np.matmul((I - np.linalg.inv((I + (l ** 2) * (D2.T * D2)).toarray())), y).A[0]
		return y_stat

	def uses_nir(self):
		return False
		
	@staticmethod
	def moving_average(values, window_size):
		m_average = np.zeros([len(values)])
		left, right = (0, window_size)
		for i in range(len(values)):
			m_average[i] = np.mean(values[i - left : i + right])
			if (left + 1 < right) or (i + right == len(values)):
				left += 1
				right -= 1

		return m_average

	@staticmethod
	def get_fft(y, frame_rate=30):
		sample_rate = 1.0 / float(frame_rate)
		sample_count = len(y)

		yf = np.fft.fft(y)
		xf = np.linspace(0.0, 1.0 / (2.0 * sample_rate), sample_count // 2)

		return xf, 2.0 / sample_count * np.abs(yf[0 : sample_count // 2])

	# https://gitlab.idiap.ch/bob/bob.rppg.base/blob/master/bob/rppg/base/utils.py
	@staticmethod
	def build_bandpass_filter(fs, order, min_freq=0.6, max_freq=4.0):
		from scipy.signal import firwin 
		
		nyq = fs / 2.0
		numtaps = order + 1
		return firwin(numtaps, [min_freq/nyq, max_freq/nyq], pass_zero=False)

	@staticmethod
	def bandpass_filter(data, frame_rate=30, min_freq=0.6, max_freq=4.0, order=64):
		from scipy.signal import filtfilt
		return filtfilt(DefaultStrategy.build_bandpass_filter(fs=float(frame_rate),
													order=order,
													min_freq=min_freq,
													max_freq=max_freq),
						np.array([1]), 
						data)